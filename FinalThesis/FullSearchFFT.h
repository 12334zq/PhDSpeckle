/*
Full search method
*/
#pragma once
#include "FullSearch.h"

class FullSearchFFT : public FullSearch
{
public:
	explicit FullSearchFFT(const Mat& first, int metric, double templRatio = 0.5, double maxShift = 0.1) : FullSearch("FullFFT", first, metric, templRatio, maxShift)
	{
		CV_Assert(metric < 6);
	}

	/**
	Get displacement with sub-pixel accuracy using full search in frequency domain
	@param frame		next frame
	@return				displacement with respect to previous frame
	*/
	Point3f getDisplacement(const Mat& frame) override
	{
		//calculate map using chosen similarity measure
		Mat result = mMetric->getMapFreq(frame(mSearchROI), mTemplate);

		//copy frame to template
		frame(mTemplateROI).copyTo(mTemplate);

		return getBestLoc(result);
	}
};
