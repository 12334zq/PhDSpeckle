#pragma once
#include "opencv2/xfeatures2d.hpp"
#include "MyFeature2D.h"

using namespace std;
using namespace cv;

class MyStarDetector : public MyFeature2D
{
protected:
	cv::Ptr<xfeatures2d::StarDetector> mFeature2D;

public:

	MyStarDetector() : MyFeature2D()
	{
		mFeature2D = xfeatures2d::StarDetector::create(10, 30);
	}

	int adjustSpecialized(int maxFeatures, const cv::Mat& image) override
	{
		vector<KeyPoint> keyPoints;
		int prevKeyPointsSize = 0;
		for (int i = 1; i < 255; ++i)
		{
			mFeature2D = xfeatures2d::StarDetector::create(10, i);
			detect(image, keyPoints);

			int keyPointsSize = keyPoints.size();
			if (keyPointsSize <= maxFeatures)
			{
				if (abs(keyPointsSize - maxFeatures) > abs(prevKeyPointsSize - maxFeatures))
				{
					mFeature2D = xfeatures2d::StarDetector::create(10, i);
					return prevKeyPointsSize;
				}

				return keyPointsSize;
			}
			prevKeyPointsSize = keyPointsSize;
			keyPoints.clear();
		}
		return -1;
	}

	cv::String getName() const override
	{
		return "CenSurE";
	}

	bool hasDescriptor() const override
	{
		return mFeature2D->descriptorSize() != 0;
	}

	void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints) const override
	{
		mFeature2D->detect(image, keypoints, mMask);
	}

	void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const override
	{
		mFeature2D->compute(image, keypoints, descriptors);
	}

	void detectAndCompute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const override
	{
		mFeature2D->detectAndCompute(image, mMask, keypoints, descriptors);
	}

	int defaultNorm() const override
	{
		return mFeature2D->defaultNorm();
	}
};
