#pragma once
#include "MatchingMethod.h"
#include "Metric.h"
#include <opencv2/imgproc.hpp>

class LowResolutionPruning : public MatchingMethod
{
	int mLayers; /**< Number of pyramid layers */
	double mPyrRatio; /**< Ratio between consecutive pyramid layers (usually > 1) */
	vector<double> mPyrFactors; /**< Scaling factors between consecutive pyramid layers (usually < 1) */
	vector<Mat> mPyramid; /**< Search area image pyramid */
	vector<Mat> mTemplates; /**< Template area image pyramid */

	/**
	Generate pyramid of search area
	@param img			input image
	*/
	void updatePyramid(const Mat& img)
	{
		img(mSearchROI).copyTo(mPyramid[0]);

		for (int i = 1; i <= mLayers; i++)
		{
			//resize with interpolation to remove high freqs
			double f = mPyrFactors[i];
			resize(img(mSearchROI), mPyramid[i], Size(), f, f);
		}
	}

	/**
	Generate pyramid of template area using search area
	@param img			input image
	*/
	void updateTemplates(const Mat& img)
	{
		img(mTemplateROI).copyTo(mTemplates[0]);

		for (int i = 1; i <= mLayers; i++)
		{
			//resize with interpolation to remove high freqs
			double f = mPyrFactors[i];
			resize(img(mTemplateROI), mTemplates[i], Size(), f, f);
		}
	}

public:
	explicit LowResolutionPruning(const Mat& first, int metric, double templRatio, double maxShift, int layers, double ratio = 2.0)
		: MatchingMethod("LowResolutionPruning", first, metric, templRatio, maxShift), mLayers(layers), mPyrRatio(ratio), mPyramid(vector<Mat>(layers + 1)), mTemplates(vector<Mat>(layers + 1))
	{
		//generate vector of ratios
		for (int i = 0; i <= mLayers; i++)
			mPyrFactors.push_back(pow(ratio, -i));

		//limit number of layers if too many
		double pyrFactor = mPyrFactors[mLayers];
		const int tempMinDim = cv::min(mTemplateROI.width, mTemplateROI.height);
		const int searchMinDim = cv::min(mSearchROI.width, mSearchROI.height);
		CV_Assert(searchMinDim > tempMinDim);

		double tempMin = tempMinDim * pyrFactor;

		bool changed = false;
		//check template minimum size
		double limitSize = 3.0;
		if (tempMin < limitSize)
		{
			mLayers = floor(log(tempMinDim / limitSize) / log(ratio));
			CV_Assert(mLayers >= 0);
			mPyrFactors.resize(mLayers + 1);
			pyrFactor = mPyrFactors[mLayers];
			changed = true;
		}

		//check minimum difference between search and template
		limitSize = 2.0;
		tempMin = tempMinDim * pyrFactor;
		if (searchMinDim * pyrFactor < tempMin + limitSize)
		{
			mLayers = floor(log((searchMinDim - tempMinDim) / limitSize) / log(ratio));
			CV_Assert(mLayers >= 0);
			mPyrFactors.resize(mLayers + 1);
			changed = true;
		}

		if (changed)
		{
			cout << "Too many layers given. Limited to " << mLayers << " layers\n";
			mPyramid.resize(mLayers + 1);
			mTemplates.resize(mLayers + 1);
		}

		updatePyramid(first);
		updateTemplates(first);
	}

	~LowResolutionPruning(){}

	/**
	Get displacement with sub-pixel accuracy using low resolution pruning with pyramids
	@param frame		next frame
	@return				displacement with respect to previous frame
	*/
	Point3f getDisplacement(const Mat& frame) override
	{
		/*
		Full-search on the lowest level of pyramid
		*/
		updatePyramid(frame);
		mMetric->reloadCache(mTemplates[mLayers]);
		Mat result = mMetric->getMapSpatial(mPyramid[mLayers], mTemplates[mLayers]);
		Point bestLoc = mMetric->findBestLoc(result);

		if (mDrawResult)
		{
			result.copyTo(mResultImg);
		}

		/*
		Coarse to fine
		*/
		Point tempBestLoc;
		for (int i = mLayers - 1; i >= 0; i--)
		{
			//scale bestLoc to lower pyramid level
			bestLoc *= mPyrRatio;

			//set next layer 3x3 search window ROI depending on bestLoc 
			Point ROIshift = Point(-1, -1); //default ROI shift
			if (bestLoc.x == 0) ROIshift += Point(1, 0);
			if (bestLoc.y == 0) ROIshift += Point(0, 1);
			Point ROItl = bestLoc + ROIshift; //ROI top-left corner position for 3x3 search window

			//search only in 3x3 search window around bestLoc
			mMetric->reloadCache(mTemplates[i]);
			result = mMetric->getMapSpatial(mPyramid[i](Rect(ROItl, mTemplates[i].size() + Size(2, 2))), mTemplates[i]);
			tempBestLoc = mMetric->findBestLoc(result);
			bestLoc += tempBestLoc + ROIshift;
		}

		/*
		Find sub-pixel accuracy
		*/
		int margin = mSubPixelEstimator->getMargin();
		int peakX = tempBestLoc.x;
		int peakY = tempBestLoc.y;

		//fill missing correlation values needed for sub-pixel estimation
		if (margin > 1 || tempBestLoc != Point(1, 1))
		{
			//expand search window
			int top = margin - tempBestLoc.y; if (top < 0) top = 0;
			int bottom = -result.rows + 1 + tempBestLoc.y + margin; if (bottom < 0) bottom = 0;
			int left = margin - tempBestLoc.x; if (left < 0) left = 0;
			int right = -result.cols + 1 + tempBestLoc.x + margin; if (right < 0) right = 0; 
			copyMakeBorder(result, result, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));

			//cache
			Size tempSize = mTemplates[0].size();
			Rect ROI = Rect(bestLoc, tempSize);
			peakX += left;
			peakY += top;

			//calculate only necessary correlations
			for (int i = 0; i < top; i++)
				result.at<float>(i, peakX) = mMetric->calculate(mPyramid[0](ROI + Point(0, -peakY + i)), mTemplates[0]);
			for (int i = 0; i < bottom; i++)
				result.at<float>(result.rows - i - 1, peakX) = mMetric->calculate(mPyramid[0](ROI + Point(0, result.rows - 1 - peakY - i)), mTemplates[0]);
			for (int i = 0; i < left; i++)
				result.at<float>(peakY, i) = mMetric->calculate(mPyramid[0](ROI + Point(-peakX + i, 0)), mTemplates[0]);
			for (int i = 0; i < right; i++)
				result.at<float>(peakY, result.cols - i - 1) = mMetric->calculate(mPyramid[0](ROI + Point(result.cols - 1 - peakX - i, 0)), mTemplates[0]);
		}

		Point2f subPix = mSubPixelEstimator->estimate(result, Point(peakX, peakY));

		//shift to get displacement
		bestLoc += mSearchROI.tl() - mTemplateROI.tl();

		updateTemplates(frame);

		return Point3f(bestLoc) + Point3f(subPix);
	}
};
