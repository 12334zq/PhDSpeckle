#pragma once
#include <opencv2/xfeatures2d.hpp>
#include "MyFeature2D.h"

using namespace std;
using namespace cv;

class MySIFT : public MyFeature2D
{
protected:
	cv::Ptr<cv::xfeatures2d::SIFT> mFeature2D;

public:

	MySIFT() : MyFeature2D()
	{
		mFeature2D = xfeatures2d::SIFT::create(0, 2, 0.01, 10, 1.2);
	}

	int adjustSpecialized(int maxFeatures, const cv::Mat& image) override
	{
		vector<KeyPoint> keyPoints;
		int prevKeyPointsSize = 0;
		for (int i = maxFeatures; i < maxFeatures * 10; ++i)
		{
			mFeature2D = xfeatures2d::SIFT::create(i, 2, 0.01, 10, 1.2);
			detect(image, keyPoints);

			int keyPointsSize = keyPoints.size();
			if (keyPointsSize >= maxFeatures)
			{
				if (abs(keyPointsSize - maxFeatures) > abs(prevKeyPointsSize - maxFeatures))
				{
					mFeature2D = xfeatures2d::SIFT::create(i, 2, 0.01, 10, 1.2);
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
		return "SIFT";
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
