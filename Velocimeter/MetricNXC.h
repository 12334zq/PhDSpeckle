#pragma once
#include "SimilarityMetric.h"

class MetricNXC : public SimilarityMetric
{
private:
	double mTempDot;

public:
	MetricNXC() : SimilarityMetric("NXC", NXC), mTempDot(-1) {}

	/**
	Calculate similarity value for two images of the same size
	@param img			image
	@param temp			template
	@return				similarity value
	*/
	float calculate(const Mat& img, const Mat& temp) override{
		
		//divison by 0 handling
		double denom = sqrt(img.dot(img)) * mTempDot;
		if (denom < DBL_EPSILON) return 0;

		return temp.dot(img) / denom;
	}

	void reloadCache(const Mat& temp) override { mTempDot = sqrt(temp.dot(temp)); }
};