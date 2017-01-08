#pragma once
#include "MyFAST.h"

class MySFAST : public MyFAST
{
public:

	MySFAST() : MyFAST()
	{
		mFeature2D->setNonmaxSuppression(true);
	}

	cv::String getName() const override
	{
		return "sFAST";
	}
};
