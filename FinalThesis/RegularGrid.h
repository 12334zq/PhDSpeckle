#ifndef REGULARGRID_H
#define REGULARGRID_H

#include <opencv2/features2d.hpp>

/**
Class derived from OpenCV Feature2D
Instead of detecting features it only creates a constant grid of points
*/
class RegularGrid : public cv ::Feature2D
{
	int mNumberOfGridRows;

public:
	RegularGrid(int rows) : mNumberOfGridRows(rows)
	{}

	void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask) override
	{
		bool useMask = false;
		cv::Mat maskMat = mask.getMat();
		if (!maskMat.empty()) useMask = true;

		int numberOfGridCols = cvRound(mNumberOfGridRows * image.cols() / static_cast<double>(image.rows()));
		keypoints = std::vector<cv::KeyPoint>(numberOfGridCols*mNumberOfGridRows);

		for (auto i = 0; i < numberOfGridCols; ++i)
			for (auto j = 0; j < mNumberOfGridRows; ++j)
			{
				float x = (j + 0.5f)*image.cols() / mNumberOfGridRows;
				float y = (i + 0.5f)*image.rows() / numberOfGridCols;

				if (useMask)
				{
					if (maskMat.at<uchar>(cv::Point(x, y)) == 0)
						continue;
				}

				keypoints.emplace_back(x, y, 0.0f);
			}
	}
};

#endif