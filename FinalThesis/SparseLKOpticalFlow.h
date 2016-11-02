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
		vector<KeyPoint> keyPoints;
		mDetector->detect(img, keyPoints);
		
		int numOfKeypoints = keyPoints.size();
		if (numOfKeypoints < MIN_NUM_OF_FEATURES)
		{
			mItersWithoutUpdate = 1;
			return;
		}

		//extract points from keyPoints, faster method than KeyPoint::convert
		mPrevPoints2f.resize(numOfKeypoints);
		for (int i = 0; i < numOfKeypoints; ++i)
		{
			mPrevPoints2f[i] = keyPoints[i].pt;
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
	Mat getTransform(const Mat& img) override
	{
		Mat M(2, 3, CV_64F);
		int i, k;

		vector<uchar> status;
		vector<Point2f> pA(mPrevPoints2f), pB;

		// find the corresponding points in B
		calcOpticalFlowPyrLK(mPrevFrame, img, pA, pB, status, noArray(), Size(27,27), 4,
			TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 30, 0.01), OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);

		// leave only points with optical flow status = true
		int count = pA.size();
		for (i = 0, k = 0; i < count; i++)
			if (status[i])
			{
				if (i > k)
				{
					pA[k] = pA[i];
					pB[k] = pB[i];
				}
				k++;
			}
		count = k;
		pB.resize(count);
		pA.resize(count);

		if (mRANSAC)
		{
			if (!RANSAC(pA, pB, 0.5)) cout << "RANSAC failed!" << endl;
		}

		if (mDrawResult)
		{
			img.copyTo(mResultImg);
			cvtColor(mResultImg, mResultImg, CV_GRAY2BGR);

			for (i = 0; i < pA.size(); i++)
			{
				circle(mResultImg, pA[i], mResultImg.cols / 80, Scalar(0, 0, 255), 2);
			}
			for (i = 0; i < pB.size(); i++)
			{
				arrowedLine(mResultImg, pA[i], pB[i], Scalar(0, 255, 0));
				circle(mResultImg, pB[i], mResultImg.cols / 80, Scalar(0, 255, 0), 2);
			}
		}

		getRTMatrix(pA, pB, M);

		mPrevPoints2f = pB;

		return M;
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
		Mat transform = getTransform(img);

		mSumFeatures += mPrevPoints2f.size();

		img.copyTo(mPrevFrame);
	
		double X = transform.at<double>(0, 2);
		double Y = transform.at<double>(1, 2);
		double cosR = transform.at<double>(0, 0);
		double sinR = transform.at<double>(1, 0);
		double angle = asin(sinR) * 180 / CV_PI;

		if (mItersWithoutUpdate >= 1 || mPrevPoints2f.size() < 10)
		{
			updateFeatures(mPrevFrame);
			mItersWithoutUpdate = 0;
		}
		else
			mItersWithoutUpdate++;

		//return Point3f(X, Y, angle);

		double cx = img.cols * 0.5;
		double cy = img.rows * 0.5;
		return Point3f(X + cx * (1.0 - cosR) - cy * sinR, Y + cx * sinR + cy * (1.0 - cosR), angle);
	}

	int getFeatures() override
	{
		return mSumFeatures;
	}
};
