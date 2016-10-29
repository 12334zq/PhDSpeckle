#pragma once
#include "MyGFTT.h"

class MyHarris : public MyGFTT
{
public:

	MyHarris() : MyGFTT()
	{
		mFeature2D->setHarrisDetector(true);
	}

	cv::String getName() const override
	{
		return "Harris";
	}
};
