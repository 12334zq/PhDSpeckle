#pragma once
#include "FullSearchFFT.h"
#include "FullSearchSpatial.h"
#include "LowResolutionPruning.h"
#include "ModifiedSpiralSearch.h"
#include "SparseOpticalFlow.h"
#include "FeatureMatching.h"

struct MethodParams
{
	int metric = Metric::NXC; /**< Metric to use */
	double templRatio = 0.7; /**< Template to image ratio */
	double maxShift = 0.1; /**< Maximum expected image shift in the next frame */
	int layers = 3; /**< Number of layers for low resolution pruning and optical flow */
	String detector = "ORB"; /**< Name of the feature detector/descriptor */
	int maxFeatures = 200;
	int RANSAC = 1; /**< 0 - all points, 1 - RANSAC */
	String matcher = "FlannBased"; /**< Name of the descriptor detector */
};

class MethodsFactory
{
public:
	/**
	Create and return pointer to specified method object using parameters struct and the first frame from source
	@param method			number of method to use (see Method::METHODS enum)
	@param first			the first frame from the source
	@param params			method parameters (see MethodParams struct)
	@return					smart OpenCV pointer to initialized method object
	*/
	static Ptr<Method> getMethod(int method, const Mat& first, MethodParams params){

		switch (method)
		{
		case Method::FULL_FFT: return new FullSearchFFT(first, params.metric, params.templRatio, params.maxShift);
		case Method::FULL_SPATIAL: return new FullSearchSpatial(first, params.metric, params.templRatio, params.maxShift);
		case Method::LRP : return new LowResolutionPruning(first, params.metric, params.templRatio, params.maxShift, params.layers, 2.0);
		case Method::SPIRAL : return new ModifiedSpiralSearch(first, params.metric, params.templRatio, params.maxShift);
		case Method::OPTICAL_FLOW: return new SparseOpticalFlow(first, params.detector, params.maxFeatures, params.RANSAC, params.layers);
		case Method::FEATURE_MATCHING : return new FeatureMatching(first, params.detector, params.matcher, params.maxFeatures, params.RANSAC);
		default: return new FullSearchSpatial(first, params.metric, params.templRatio, params.maxShift);
		}

	}
};
