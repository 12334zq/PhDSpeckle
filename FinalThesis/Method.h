/** @file */
#pragma once
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/mat.hpp"

using namespace cv;


/**
Base class for every method in the system.
*/
class Method
{
	String mMethodName; /**< Method name */

protected:
	bool mDrawResult; /**< Results drawing flag */
	Mat mResultImg; /**< Method resulting image */

	/**
	Add segment to method name
	@param segment			segment to add
	*/
	void addToName(const String& segment) { mMethodName = mMethodName + "_" + segment; }

public:
	/**
	@enum enum for easier methods selection
	*/
	enum
	{
		FULL_FFT, FULL_SPATIAL, LRP, SPIRAL, OPTICAL_FLOW, FEATURE_MATCHING
	};

	explicit Method(const String& name) : mMethodName(name), mDrawResult(false), mResultImg{} {}

	virtual ~Method() {}

	/**
	Get method name
	@return				string with method name
	*/
	String getName() const { return mMethodName; }

	/**
	Switch results drawing on/off
	*/
	void drawingOnOff() { mDrawResult = !mDrawResult; }

	/**
	Get result image
	@return				Mat with result
	*/
	Mat getResultImg() const { return mResultImg;  }

	/**
	Virtual method derived by every inherited class. 
	Get displacement with sub-pixel accuracy
	@param img			next frame
	@return				displacement with respect to previous frame
	*/
	virtual Point3f getDisplacement(const Mat& img) = 0;

	virtual int getFeatures() {return 0;} 
};
