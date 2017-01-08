/*
Partial search
Spiral search with prediction
*/
#pragma once
#include "MatchingMethod.h"
#include <iostream>
#include <opencv2/tracking/feature.hpp>

class ModifiedSpiralSearch : public MatchingMethod
{
	Point mGuess; /**< Previously calculated shift */
	double mPredThresh; /**< Threshold to accept prediction */
	double mPeakThresh; /**< Threshold to accept peak */
	Mat mCache; /**< Container for correlation values */
	int mCacheCols, mCacheRows;
	Point mROItl;

	//directions of template movement
	const Point UP = Point(0, -1);
	const Point RIGHT = Point(1, 0);
	const Point DOWN = Point(0, 1);
	const Point LEFT = Point(-1, 0);
	const vector<Point> DIRECTIONS = vector<Point>{ UP, RIGHT, DOWN, LEFT };

	/**
	Set constant thresholds for peak finding process.
	TODO Adaptive algorithm for defining thresholds
	@param metric				metric number
	*/
	void setThresholds(int metric)
	{
		if (metric == Metric::ZXC)
		{
			mPredThresh = mTemplateROI.area() * 400;
			mPeakThresh = mTemplateROI.area() * 300;
		}
		else if (metric == Metric::MAD)
		{
			mPredThresh = 16;
			mPeakThresh = 13;
		}
		else if (metric == Metric::ZNXC)
		{
			mPredThresh = 0.6;
			mPeakThresh = 0.7;
		}
		else if (metric == Metric::NSSD)
		{
			mPredThresh = 0.4;
			mPeakThresh = 0.3;
		}
		else if (metric == Metric::NXC)
		{
			mPredThresh = 0.8;
			mPeakThresh = 0.9;
		}
		else if (metric == Metric::SAD)
		{
			mPredThresh = mTemplateROI.area() * 46;
			mPeakThresh = mTemplateROI.area() * 38;
		}
		else if (metric == Metric::SSD)
		{
			mPredThresh = mTemplateROI.area() * mTemplateROI.area() * 60 * 60;
			mPeakThresh = mTemplateROI.area() * mTemplateROI.area() * 50 * 50;
		}
		else if (metric == Metric::XC)
		{
			mPredThresh = mTemplateROI.area() * 1325;
			mPeakThresh = mTemplateROI.area() * 1500;
		}
	}

	/**
	Update shift and all parameters using spiral motion routine. Execute procedure "iters" times
	*/
	void iterateSpiral(Point& shift, int* countToLevel, int* spiralLevel, int* direction, int* countToTwo, int iters) const
	{
		for (int i = 0; i < iters; i++)
		{
			shift += DIRECTIONS[*direction];
			//change direction of search
			if (++*countToLevel == *spiralLevel)
			{
				//change direction
				if (++*direction == 4) *direction = 0;

				//change spiral level every 2 steps
				if (++*countToTwo == 2)
				{
					*countToTwo = 0;
					++*spiralLevel;
				}
				*countToLevel = 0;
			}
		}
	}

public:
	explicit ModifiedSpiralSearch(const Mat& first, int metric, double templRatio = 0.5, double maxShift = 0.1) : MatchingMethod("SpiralSearch", first, metric, templRatio, maxShift), mGuess(0)
	{
		setThresholds(metric);
		mCache = Mat(mSearchROI.size() - mTemplateROI.size() + Size(1, 1), CV_32F, Scalar(-FLT_MAX));

		//leave limit for sub-pixel calculation
		mCacheCols = mSearchROI.size().width - mTemplateROI.size().width + 1;
		mCacheRows = mSearchROI.size().height - mTemplateROI.size().height + 1;
		mROItl = mMaxTranslation + Point(1,1);
	}

	/**
	Get displacement with sub-pixel accuracy using spiral search
	@param frame		next frame
	@return				displacement with respect to previous frame
	*/
	Point3f getDisplacement(const Mat& img) override
	{
		//reset cache
		mCache = Scalar(-FLT_MAX);

		//next translation guessing
		Point shift(mGuess);

		//correlation value for predicted shift
		float value = mCache.at<float>(mROItl + shift) = mMetric->calculate(img(mTemplateROI + shift), mTemplate);

		//main loop
		int spiralLevel = 1; // actual spiral level
		int direction = 0; // actual direction of movement from DIRECTIONS vector
		int status = -2; // -2 if no peak found, -1 if peak found
		int countToTwo = 0;
		int countToLevel = 0;

		//1. default stop condition of the loop: spiralLevel too high
		const int maxTL = 2 * max(mROItl.x + shift.x, mROItl.y + shift.y) - 1;
		const int maxBR = 2 * max(mCacheCols - mROItl.x - shift.x, mCacheRows - mROItl.y - shift.y);
		const int maxSpiralLevel = max(maxTL, maxBR);
		//2. stop condition inside the loop: correlation peak found
		do
		{
			if (!mMetric->isBetter(value, mPeakThresh)) //phase 1: spiral search for a peak neighbourhood
			{
				/*
				Reduction in search points based on similarity threshold
				*/
				int step = 2; //default step for moderate correlation
				if (!mMetric->isBetter(value, mPredThresh)) step = 3; //larger step for poor correlation
				iterateSpiral(shift, &countToLevel, &spiralLevel, &direction, &countToTwo, step);

				//immediately skip loop iteration if shift not in the search ROI range
				const Point target = mROItl + shift;
				if (target.x < 0 || target.y < 0 || target.x > mCacheCols - 1 || target.y > mCacheRows - 1)
					continue;

				//calculate correlation value for next shift
				value = mCache.at<float>(target) = mMetric->calculate(img(mTemplateROI + shift), mTemplate);
			}
			else //phase 2: detailed cross search for the actual peak
			{
				//find maximum value in cross-neighbourhood of actual shift
				status = -1;
				for (int i = 0; i < 4; i++)
				{
					//immediately skip loop iteration if shift not in the search ROI range
					const Point nextShift = shift + DIRECTIONS[i];
					const Point target = mROItl + nextShift;
					if (target.x < 0 || target.y < 0 || target.x > mCacheCols - 1 || target.y > mCacheRows - 1)
					{
						status = -2;
						continue;
					}

					//check if value for shift is in cache - shift in cache cannot be peak
					if (mCache.at<float>(target) < -INT_MAX)
					{
						double corr = mCache.at<float>(target) = mMetric->calculate(img(mTemplateROI + nextShift), mTemplate);
						//change search status if actual value is not a peak
						if (mMetric->isBetter(corr, value))
						{
							value = corr;
							status = i;
						}
					}
				}

				/*
				Early jump-out procedure
				peak found if value is the heighest in the neighbourhood
				*/
				if (status < 0) break;

				//shift to the highest neighbour
				shift += DIRECTIONS[status];
			}
		} while (spiralLevel < maxSpiralLevel);

		//check searching status
		if (status != -1)
		{
			cout << "WARNING: Border of disparity map reached!" << endl;
			return Point3f(mGuess);
		}

		//fill cache with values needed for sub-pixel estimation
		if (mSubPixelEstimator->getMargin() == 2)
		{
			for (int i = 0; i < 4; i++)
			{
				const Point nextShift = shift + 2 * DIRECTIONS[i];
				const Point target = mROItl + nextShift;
				if (target.x < 0 || target.y < 0 || target.x > mCacheCols - 1 || target.y > mCacheRows - 1)
				{
					cout << "WARNING: Border of disparity map reached!" << endl;
					return Point3f(mGuess);
				}
				if (mCache.at<float>(target) < -INT_MAX)
				{
					mCache.at<float>(target) = mMetric->calculate(img(mTemplateROI + nextShift), mTemplate);
				}
			}
		}

		if (mDrawResult)
		{
			Mat temp;
			mCache.forEach<float>([](float& p, const int* pos) -> void {if (p < -INT_MAX) p = -1; });
			normalize(mCache, temp, 0, 255, NORM_MINMAX);
			temp.convertTo(temp, CV_8UC1);
			temp.copyTo(mResultImg);
		}

		//update template
		img(mTemplateROI).copyTo(mTemplate);
		mMetric->reloadCache(mTemplate);

		//update guessing
		mGuess = shift;

		//find sub-pixel accuracy
		Point2f subPix = mSubPixelEstimator->estimate(mCache, mROItl + shift);

		return Point3f(Point2f(shift) + subPix);
	}
};
