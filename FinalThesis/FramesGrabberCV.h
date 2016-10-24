#pragma once
#include "FramesGrabber.h"
#include "opencv2/videoio/videoio.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

class FramesGrabberCV : public FramesGrabber
{
	VideoCapture mVidCap; /**< Camera or video connection */
	String mSourceType; /**< Name of frame source: Camera, Video or Sequence */

public:
	/**
	Open given camera
	@param cam_id			id of camera to open
	*/
	FramesGrabberCV(int cam_id) : FramesGrabber()
	{
		//check for camera availaibilty
		if (!mVidCap.open(cam_id))
		{
			CV_Error(CV_StsBadArg, String(mSourceType + " " + to_string(cam_id) + " not found!\n"));
		}
		else
		{
			mSourceType = "Camera";
			cout << "FramesGrabberCV: " << mSourceType << " ready\n";
			mFPS = -1;
		}
	}

	/**
	Open given videofile or images sequence
	@param videoname		path of videofile to open
	*/
	FramesGrabberCV(const String& videoname) : FramesGrabber()
	{
		//check for video/images availability
		if (videoname.find(".jpg") == string::npos &&
			videoname.find(".png") == string::npos &&
			videoname.find(".bmp") == string::npos &&
			videoname.find(".gif") == string::npos)
		{
			if (!mVidCap.open(videoname, CAP_FFMPEG))
				CV_Error(CV_StsBadArg, String(mSourceType + " " + videoname + " not found!\n"));
			else
			{
				mSourceType = "Video";
				cout << "FramesGrabberCV: " << mSourceType << " " << videoname << " ready\n";
				mFPS = mVidCap.get(CAP_PROP_FPS);
			}
		}
		else
		{
			if (!mVidCap.open(videoname, CAP_IMAGES))
				CV_Error(CV_StsBadArg, String(mSourceType + " " + videoname + " not found!\n"));
			else
			{
				mSourceType = "Sequence";
				cout << "FramesGrabberCV: " << mSourceType << " " << videoname << " ready\n";
			}
		}

	}

	~FramesGrabberCV()
	{
		if (mVidCap.isOpened())
		mVidCap.release();
	}

	/**
	Open given videofile or images sequence
	@param frame		reference to container for next frame
	@return				acquisition status (false - acquisition stopped / true - acquisition in progress)
	*/
	bool acquire(Mat& output) override
	{
		if (!mVidCap.read(output))
		{
			cout << mSourceType << " stopped. Press any key\n";
			return false;
		}

		if (output.type() != CV_8UC1) cvtColor(output, output, CV_BGR2GRAY);

		mFrameNumber++;
		if (mFrameNumber == 1)
		{
			setSize(output.size());
			return true;
		}

		if (output.size() != mSize)
			CV_Error(Error::StsUnmatchedSizes, "Consecutive images must have the same size");
		else
			return true;

		return false;
	}
};
