#pragma once
#include "DissimilarityMetric.h"

class MetricNSSD : public DissimilarityMetric
{
private:
	double mTempDot; /**< Keep dot product of template in cache */

public:
	MetricNSSD() : DissimilarityMetric("NSSD", NSSD), mTempDot(0) {}

	/**
	Calculate similarity value for two images of the same size
	@param img			image
	@param temp			template
	@return				similarity value
	*/
	float calculate(const Mat& img, const Mat& temp) override{
		//divison by 0 handling
		double denom = sqrt(img.dot(img));
		if (mTempDot <= FLT_EPSILON || denom <= FLT_EPSILON)
		{
			if (denom > FLT_EPSILON || mTempDot > FLT_EPSILON) return 1;
			return 0;
		}

		return norm(temp, img, NORM_L2SQR) / (denom * mTempDot);
	}

	void reloadCache(const Mat& temp) override{
		mTempDot = sqrt(temp.dot(temp));
	}
};
