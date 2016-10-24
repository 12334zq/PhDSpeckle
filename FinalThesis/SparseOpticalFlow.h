#pragma once
#include "FeaturesMethod.h"
#include <fstream>

/**
Class for sparse optical flow
*/
class SparseOpticalFlow : public FeaturesMethod
{
	vector<Point2f> mPrevPoints2f; /**< Previously detected points */
	int mLayers; /**< Number of layers in optical flow algorithm */
	double sumDetectTime = 0;
	int frames = 0;
	int mSumFeatures = 0;
	int mIter = 0;

	/**
	Calculate and update features vector
	@param img			input image
	*/
	void updateFeatures(const Mat& img)
	{
		//detect keypoints in image
		std::vector<KeyPoint> keyPoints;

		mDetector->detect(img, keyPoints, mDetectorMask);
		
		int numOfKeypoints = keyPoints.size();

		//stop if too low number of matches to achieve reasonable result
		if (numOfKeypoints < 9)
		{
			mIter = 1;
			return;
		}

		mMaxFeatures = numOfKeypoints;

		//convert keyPoints to points
		mPrevPoints2f.resize(numOfKeypoints);
		for (size_t i = 0; i < numOfKeypoints; i++)
		{
			mPrevPoints2f[i] = keyPoints[i].pt;
		}

		//add subpixel accuracy if needed
		if (mDetectorName != "SURF" && mDetectorName != "U-SURF")
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

		if (mEstimationType == 1)
		{
			bool result = RANSAC(pA, pB, 0.5);
			if (!result) cout << "RANSAC failed!" << endl;
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

		getRTMatrix(pA, pB, pA.size(), M);

		mPrevPoints2f = pB;

		return M;
	}


public:
	SparseOpticalFlow(const Mat& first, const String& detector, int maxFeatures, int estimation = 0, int layers = 4) : FeaturesMethod("OpticalFlow", first, detector, maxFeatures, estimation), mLayers(layers)
	{
		//only 8-bit 1-channel supported
		if (first.type() != CV_8UC1)
			CV_Error(Error::StsUnsupportedFormat, "Input images must have 8UC1 type");

		KeyPoint::convert(mPrevKeypoints, mPrevPoints2f);
		updateFeatures(first);

		if (estimation == 2) addToName("_OpenCV");
	}

	~SparseOpticalFlow()
	{
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
		//don't detect new keypoints if motion is small and features are complete
		double X = transform.at<double>(0, 2);
		double Y = transform.at<double>(1, 2);
		double cosR = transform.at<double>(0, 0);
		double sinR = transform.at<double>(1, 0);
		double angle = asin(sinR) * 180 / CV_PI;

		if (mIter >= 1 || mPrevPoints2f.size() < 10)
		{
			updateFeatures(mPrevFrame);
			mIter = 0;
		}
		else
			mIter++;

		return Point3f(X, Y, angle);

		/*double cx = img.cols * 0.5;
		double cy = img.rows * 0.5;
		return Point3f(X - cx * (1.0 - cosR) + cy * sinR, Y - cx * sinR - cy * (1.0 - cosR), angle);*/
	}

	int getFeatures() override
	{
		return mSumFeatures;
	}
};
