#pragma once
#include <opencv2/video/tracking.hpp>
#include "FeaturesMethod.h"

/**
Class for sparse optical flow
*/
class SparseLKOpticalFlow : public FeaturesMethod
{
	vector<Point2f> mPrevPoints2f; /**< Previously detected points */
	int mLayers; /**< Number of layers in optical flow algorithm */
	int mSumFeatures = 0;
	int mItersWithoutUpdate = 0;

	/**
	Calculate and update features vector
	@param img			input image
	*/
	void updateFeatures(const Mat& img)
	{
		mDetector->detect(img, mPrevKeypoints);
		
		int numOfKeypoints = mPrevKeypoints.size();
		if (numOfKeypoints < MIN_NUM_OF_FEATURES)
		{
			mItersWithoutUpdate = 1;
			return;
		}

		//extract points from keyPoints, faster method than KeyPoint::convert
		mPrevPoints2f.resize(numOfKeypoints);
		for (int i = 0; i < numOfKeypoints; ++i)
		{
			mPrevPoints2f[i] = mPrevKeypoints[i].pt;
		}

		//add subpixel accuracy if needed
		if (mDetector->getName() != "SURF" && mDetector->getName() != "U-SURF")
		{
			cornerSubPix(img, mPrevPoints2f, Size(3, 3), Size(-1, -1),
				TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 30, 0.01));
		}
	}

	/**
	Find rigid transformation matrix for the next frame
	@param frame		next frame
	@return				transformation matrix
	*/
	Transform processNextFrame(const Mat& img) override
	{
		vector<uchar> status;
		vector<Point2f> pA(mPrevPoints2f), pB;

		// find the corresponding points in B
		calcOpticalFlowPyrLK(mPrevFrame, img, pA, pB, status, noArray(), Size(27,27), 4,
			TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.01), OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);

		int i = 0;
		pA.erase(remove_if(pA.begin(), pA.end(), [&](...) {return !status[i++]; }), pA.end());
		i = 0;
		pB.erase(remove_if(pB.begin(), pB.end(), [&](...) {return !status[i++]; }), pB.end());

		if (mRANSAC)
		{
			if (!RANSAC(pA, pB, 0.5)) cout << "RANSAC failed!" << endl;
		}

		if (mDrawResult)
		{
			img.copyTo(mResultImg);
			cvtColor(mResultImg, mResultImg, CV_GRAY2BGR);

			auto COLOR_RED = Scalar(0, 0, 255);
			auto COLOR_GREEN = Scalar(0, 255, 0);
			for (i = 0; i < pB.size(); ++i)
			{
				circle(mResultImg, pA[i], mResultImg.cols / 80, COLOR_RED, 2);
				arrowedLine(mResultImg, pA[i], pB[i], COLOR_GREEN);
				circle(mResultImg, pB[i], mResultImg.cols / 80, COLOR_GREEN, 2);
			}
		}

		mPrevPoints2f = pB;

		return getTransform(pA, pB);
	}


public:
	SparseLKOpticalFlow(const Mat& first, int detector, int maxFeatures, bool RANSAC = false, int layers = 4)
		: FeaturesMethod("OpticalFlow", first, detector, maxFeatures, RANSAC), mLayers(layers)
	{
		//only 8-bit 1-channel supported
		if (first.type() != CV_8UC1)
			CV_Error(Error::StsUnsupportedFormat, "Input images must have 8UC1 type");

		KeyPoint::convert(mPrevKeypoints, mPrevPoints2f);
		updateFeatures(first);
	}

	/**
	Get displacement with sub-pixel accuracy using optical flow algorithm
	@param frame		next frame
	@return				displacement with respect to previous frame
	*/
	Point3f getDisplacement(const Mat& img) override
	{
		Transform transform = processNextFrame(img);

		mSumFeatures += mPrevPoints2f.size();

		img.copyTo(mPrevFrame);

		if (mItersWithoutUpdate >= 1 || mPrevPoints2f.size() < 10)
		{
			updateFeatures(mPrevFrame);
			mItersWithoutUpdate = 0;
		}
		else
			mItersWithoutUpdate++;

		return Point3f(transform.tx, transform.ty, transform.angle);
	}

	int getFeatures() override
	{
		return mSumFeatures;
	}
};
