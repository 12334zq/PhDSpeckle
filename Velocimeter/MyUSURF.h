#pragma once
#include "MySURF.h"

class MyUSURF : public MySURF
{
public:

	MyUSURF() : MySURF()
	{
		mFeature2D->setUpright(true);
	}

	cv::String getName() const override
	{
		return "uSURF";
	}
};
