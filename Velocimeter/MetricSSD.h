#pragma once
#include "DissimilarityMetric.h"

class MetricSSD : public DissimilarityMetric
{
public:
	MetricSSD() : DissimilarityMetric("SSD", SSD) {}

	/**
	Calculate similarity value for two images of the same size
	@param img			image
	@param temp			template
	@return				similarity value
	*/
	float calculate(const Mat& img, const Mat& temp) override{
		return cv::norm(temp, img, NORM_L2SQR);
	}

};
