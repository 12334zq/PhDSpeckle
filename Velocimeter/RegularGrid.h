#ifndef REGULARGRID_H
#define REGULARGRID_H

#include <opencv2/features2d.hpp>
#include "MyFeature2D.h"

using namespace cv;
using namespace std;

/**
Class derived from OpenCV Feature2D
Instead of detecting features it only creates a constant grid of points
*/
class RegularGrid : public MyFeature2D
{
	vector<KeyPoint> mKeyPointsGrid{};

public:
	RegularGrid() {}

	int adjustSpecialized(int maxFeatures, const cv::Mat& image) override
	{
		int pixelsInImage = image.cols * image.rows;
		int pixelsInMask = countNonZero(mMask);
		if (mMask.empty()) pixelsInMask = pixelsInImage;
		double imageToMaskRatio = pixelsInImage / static_cast<double>(pixelsInMask);
		int maxFeaturesPerImage = static_cast<int>(maxFeatures * imageToMaskRatio);

		double aspectRatio = image.cols / static_cast<double>(image.rows);
		int gridRows = static_cast<int>(sqrt(maxFeaturesPerImage / aspectRatio));
		int gridCols = static_cast<int>(gridRows * aspectRatio);

		mKeyPointsGrid.clear();
		double horizontalStep = image.cols / static_cast<double>(gridCols);
		double verticalStep = image.rows / static_cast<double>(gridRows);
		for (auto i = 1; i < gridCols - 1; ++i)
			for (auto j = 1; j < gridRows - 1; ++j)
			{
				mKeyPointsGrid.emplace_back(i * horizontalStep, j * verticalStep, 0.0f);
			}

		KeyPointsFilter::runByPixelsMask(mKeyPointsGrid, mMask);

		return mKeyPointsGrid.size();
	}

	cv::String getName() const override
	{
		return "Grid";
	}

	bool hasDescriptor() const override
	{
		return false;
	}

	void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints) const override
	{
		keypoints = mKeyPointsGrid;
	}

	void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const override
	{
		detectAndCompute(image, keypoints, descriptors);
	}

	void detectAndCompute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const override
	{
		CV_Error(cv::Error::StsNotImplemented, "");
	}

	int defaultNorm() const override
	{
		return cv::NORM_L2;
	}
};

#endif