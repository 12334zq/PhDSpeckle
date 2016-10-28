#pragma once
#include "opencv2/features2d.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "RegularGrid.h"
#include <map>

//wrapper around Feature2D adding name and mask
class MyFeature2D
{
public: enum {
	Grid, AGAST, sAGAST, BRISK, FAST, sFAST, GFTT, Harris, ORB, SURF, uSURF, SIFT
};

private:
	cv::Ptr<cv::Feature2D> mDetector;
	cv::String mName;
	cv::Mat mMask{};

public:
	MyFeature2D(int detector, int maxFeatures, const Mat& image, const cv::Mat& mask = cv::Mat()) : mMask(mask)
	{
		const map<int, cv::String> mDetectorNames{
			{ Grid, "Grid" },
			{ AGAST, "AGAST" },
			{ sAGAST, "sAGAST" },
			{ BRISK, "BRISK" },
			{ FAST, "FAST" },
			{ sFAST, "sFAST" },
			{ GFTT, "GFTT" },
			{ Harris, "Harris" },
			{ ORB, "ORB" },
			{ SURF, "SURF" },
			{ uSURF, "uSURF" },
			{ SIFT, "SIFT" }
		};
		mName = mDetectorNames.at(detector);

		cout << "Adjusting settings to achieve desired number of features...\n";
		vector<KeyPoint> keyPoints;

		if (detector == Grid)
		{
			mDetector = cv::makePtr<RegularGrid>(round(image.rows / 27.0));
		}
		else if (detector == AGAST)
		{
			for (int i = 1; i < 255; ++i)
			{
				keyPoints.clear();
				mDetector = AgastFeatureDetector::create(i, false);
				detect(image, keyPoints);
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == sAGAST)
		{
			for (int i = 1; i < 255; ++i)
			{
				keyPoints.clear();
				mDetector = AgastFeatureDetector::create(i, true);
				detect(image, keyPoints);
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == BRISK)
		{
			for (int i = 36; i < 255; ++i)
			{
				keyPoints.clear();
				mDetector = BRISK::create(i, 0);
				detect(image, keyPoints);
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == FAST)
		{
			for (int i = 1; i < 255; ++i)
			{
				keyPoints.clear();
				mDetector = FastFeatureDetector::create(i, false);
				detect(image, keyPoints);
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == sFAST)
		{
			for (int i = 1; i < 255; ++i)
			{
				keyPoints.clear();
				mDetector = FastFeatureDetector::create(i, true);
				detect(image, keyPoints);
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == GFTT)
			mDetector = GFTTDetector::create(maxFeatures, 0.01, 1);
		else if (detector == Harris)
			mDetector = GFTTDetector::create(maxFeatures, 0.01, 1, 3, true);
		else if (detector == ORB)
			mDetector = ORB::create(maxFeatures, 1.2, 1, 0);
		else if (detector == SURF)
		{
			int step = 10;
			for (int i = 300; i < 50000; i += step)
			{
				keyPoints.clear();
				mDetector = xfeatures2d::SURF::create(i, 1, 3);
				mDetector->detect(image, keyPoints);
				//cout << i << "," << mPrevKeypoints.size() << endl;
				if (keyPoints.size() < 180) step = 15;
				if (keyPoints.size() < 130) step = 25;
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == uSURF)
		{
			int step = 10;
			for (int i = 300; i < 50000; i += step)
			{
				keyPoints.clear();
				mDetector = xfeatures2d::SURF::create(i, 1, 3, false, true);
				mDetector->detect(image, keyPoints);
				//cout << i << "," << mPrevKeypoints.size() << endl;
				if (keyPoints.size() < 180) step = 15;
				if (keyPoints.size() < 130) step = 25;
				if (keyPoints.size() <= maxFeatures) break;
			}
		}
		else if (detector == SIFT)
		{
			for (int i = maxFeatures; i < maxFeatures * 10; ++i)
			{
				keyPoints.clear();
				mDetector = xfeatures2d::SIFT::create(i, 2, 0.01, 10, 1.2);
				mDetector->detect(image, keyPoints);
				if (keyPoints.size() >= maxFeatures) break;
			}
		}
	}


	void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints) const
	{
		mDetector->detect(image, keypoints, mMask);
	}

	void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray imgDescriptor) const
	{
		mDetector->compute(image, keypoints, imgDescriptor);
	}

	void detectAndCompute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const
	{
		mDetector->detectAndCompute(image, mMask, keypoints, descriptors);
	}

	cv::String getName() const
	{
		return mName;
	}

	int descriptorSize() const
	{
		return mDetector->descriptorSize();
	}

	int defaultNorm() const
	{
		return mDetector->defaultNorm();
	}
	
};

