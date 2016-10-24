#pragma once
#include "SubPixelEstimator.h"

class Gaussian3 : public SubPixelEstimator
{
public:
	/**
	Calculate sub-pixel accuracy using 3-point Gaussian peak estimation in 2D
	@param map			correlation image
	@param bestLoc		peak with pixel accuracy
	@return				estimated sub-pixel translation value
	*/
	Point2f estimate(const Mat& corr, Point bestLoc) const override
	{
		const int y = bestLoc.y;
		const int x = bestLoc.x;
		float xy2, ym1, xp1, yp1, xm1;
		try
		{
			xy2 = 2 * log(corr.at<float>(y, x));
			ym1 = log(corr.at<float>(y - 1, x));
			xp1 = log(corr.at<float>(y, x + 1));
			yp1 = log(corr.at<float>(y + 1, x));
			xm1 = log(corr.at<float>(y, x - 1));
		}
		catch (Exception ex)
		{
			return Point2f(0, 0);
		}

		const float dx = (xm1 - xp1) / (2 * (xm1 - xy2 + xp1));
		const float dy = (ym1 - yp1) / (2 * (ym1 - xy2 + yp1));

		return Point2f(dx, dy);
	}

	/**
	Get estimator id number from ESTIMATOR_TYPE enum
	@return				sub-pixel estimator number
	*/
	int getType() const override
	{
		return GAUSS3;
	}

	/**
	Get number of pixels from neighbourhood needed to calculate sub-pixel estimation.
	Margin N means (2N + 1) x (2N + 1) neighbourhood
	@return				margin value in pixels
	*/
	int getMargin() const override
	{
		return 1;
	}
};
