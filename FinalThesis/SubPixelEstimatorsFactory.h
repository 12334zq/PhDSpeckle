#pragma once
#include "Gaussian3.h"
#include "Gaussian5.h"

class SubPixelEstimatorsFactory
{
public:
	/**
	Create and return pointer to specified SubPixelEstimator object
	@param type				number of metric to use (see SubPixelEstimator::ESTIMATOR_TYPE enum)
	@return					smart OpenCV pointer to initialized SubPixelEstimator object
	*/
	static Ptr<SubPixelEstimator> getEstimator(int type){

		switch (type)

		{
		case SubPixelEstimator::GAUSS3: return new Gaussian3();
		case SubPixelEstimator::GAUSS5: return new Gaussian5();
		default: return new Gaussian3();
		}
	}
};
