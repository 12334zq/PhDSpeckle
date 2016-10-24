#pragma once
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

/**
Base class for different frame grabbers
*/
class FramesGrabber {

protected:
	double mFPS; /**< Frames per second */
	Size mSize; /**< First frame size */
	int mType; /**< First frame type */
	uint64 mFrameNumber; /**< Actual frame number */

	/**
	Set frame size
	@param size			size to set
	*/
	void setSize(Size size){ mSize = size; }

	/**
	Set frame type
	@param size			type to set
	*/
	void setType(int type){ mType = type; }

public:

	FramesGrabber() : mFPS(24.0f), mSize(Size(0,0)), mType(-1), mFrameNumber(0){}

	virtual ~FramesGrabber(){}

	/**
	Get number of frames per second
	@return				frames per second
	*/
	double getFPS() const { return mFPS; }

	/**
	Acquire next frame
	@param output	acquired frame
	@return			true if success, false otherwise
	*/
	virtual bool acquire(Mat& output) = 0;

};