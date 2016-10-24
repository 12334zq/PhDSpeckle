#pragma once
#include "SimilarityMetric.h"

class MetricZXC : public SimilarityMetric
{
private:
	Mat mZeroMeanTemp;
	Mat mZeroMeanImg;

public:
	MetricZXC() : SimilarityMetric("ZXC", ZXC), mZeroMeanTemp(), mZeroMeanImg() {}

	void reloadCache(const Mat& temp) override 
	{
		temp.convertTo(mZeroMeanTemp, CV_32F);
		mZeroMeanTemp = mZeroMeanTemp - mean(temp);
	}

	/**
	Calculate similarity value for two images of the same size
	@param img			image
	@param temp			template
	@return				similarity value
	*/
	float calculate(const Mat& img, const Mat& temp) override{

		img.convertTo(mZeroMeanImg, CV_32F);
		mZeroMeanImg = mZeroMeanImg - mean(img);

		return mZeroMeanTemp.dot(mZeroMeanImg);
	}
};
