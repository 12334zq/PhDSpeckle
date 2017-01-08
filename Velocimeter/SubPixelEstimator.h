#pragma once
#include <opencv2/core/core.hpp>

using namespace cv;

class SubPixelEstimator
{
public:
	/**
	@enum enum for easier estimator selection
	*/
	enum ESTIMATOR_TYPE
	{
		GAUSS3, GAUSS5
	};

	virtual ~SubPixelEstimator(){}

	/**
	Calculate sub-pixel accuracy
	@param map			correlation image
	@param bestLoc		peak with pixel accuracy
	@return				estimated sub-pixel translation value
	*/
	virtual Point2f estimate(const Mat& map, Point bestLoc) const { return Point2f(0); }

	/**
	Get estimator id number from ESTIMATOR_TYPE enum
	@return				sub-pixel estimator number
	*/
	virtual int getType() const = 0;

	/**
	Get number of pixels from neighbourhood needed to calculate sub-pixel estimation.
	Margin N means (2N + 1) x (2N + 1) neighbourhood
	@return				margin value in pixels
	*/
	virtual int getMargin() const = 0;
};
