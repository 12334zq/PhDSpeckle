#pragma once
#include "DissimilarityMetric.h"

class MetricMAD : public DissimilarityMetric
{
public:
	MetricMAD() : DissimilarityMetric("MAD", MAD) {}

	/**
	Calculate similarity value for two images of the same size
	@param img			image
	@param temp			template
	@return				similarity value
	*/
	float calculate(const Mat& img, const Mat& temp) override{
		return norm(temp, img, NORM_L1)/img.total();
	}
};