#pragma once
#include "MyAGAST.h"

class MySAGAST : public MyAGAST
{
public:

	MySAGAST() : MyAGAST()
	{
		mFeature2D->setNonmaxSuppression(true);
	}

	cv::String getName() const override
	{
		return "sAGAST";
	}
};
