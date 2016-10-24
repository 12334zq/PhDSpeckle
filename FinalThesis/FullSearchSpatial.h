/*
Full search method
*/
#pragma once
#include "FullSearch.h"

class FullSearchSpatial : public FullSearch
{

public:
	explicit FullSearchSpatial(const Mat& first, int metric, double templRatio = 0.5, double maxShift = 0.1) : FullSearch("FullSpatial", first, metric, templRatio, maxShift){}

	/**
	Get displacement with sub-pixel accuracy using full search in spatial domain
	@param frame		next frame
	@return				displacement with respect to previous frame 
	*/
	Point3f getDisplacement(const Mat& frame) override
	{
		//calculate map using chosen similarity measure
		Mat result = mMetric->getMapSpatial(frame(mSearchROI), mTemplate);

		//copy frame to template
		frame(mTemplateROI).copyTo(mTemplate);
		mMetric->reloadCache(mTemplate);

		return getBestLoc(result);
	}

};