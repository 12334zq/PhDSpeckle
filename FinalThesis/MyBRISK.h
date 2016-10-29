#pragma once
#include "opencv2/features2d.hpp"
#include "MyFeature2D.h"

using namespace std;
using namespace cv;

class MyBRISK : public MyFeature2D
{
protected:
	cv::Ptr<cv::BRISK> mFeature2D;

public:

	MyBRISK() : MyFeature2D()
	{
		mFeature2D = BRISK::create(30, 0);
	}

	int adjustSpecialized(int maxFeatures, const cv::Mat& image) override
	{
		vector<KeyPoint> keyPoints;
		int prevKeyPointsSize = 0;
		for (int i = 36; i < 255; ++i)
		{
			mFeature2D = BRISK::create(i);
			detect(image, keyPoints);

			int keyPointsSize = keyPoints.size();
			if (keyPointsSize <= maxFeatures)
			{
				if (abs(keyPointsSize - maxFeatures) > abs(prevKeyPointsSize - maxFeatures))
				{
					mFeature2D = BRISK::create(i - 1);
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
		return "BRISK";
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
