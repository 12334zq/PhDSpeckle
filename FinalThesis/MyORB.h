#pragma once
#include "opencv2/features2d.hpp"
#include "MyFeature2D.h"

using namespace std;
using namespace cv;

class MyORB : public MyFeature2D
{
protected:
	cv::Ptr<cv::ORB> mFeature2D;

public:

	MyORB() : MyFeature2D()
	{
		mFeature2D = ORB::create(200, 1.2, 1, 0);
	}

	int adjustSpecialized(int maxFeatures, const cv::Mat& image) override
	{
		mFeature2D->setMaxFeatures(maxFeatures);

		vector<KeyPoint> keyPoints;
		detect(image, keyPoints);

		return keyPoints.size();
	}

	cv::String getName() const override
	{
		return "ORB";
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
