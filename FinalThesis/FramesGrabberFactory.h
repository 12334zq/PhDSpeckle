#pragma once
#include "FramesGrabberSeq.h"
#include "FramesGrabberCV.h"

class FramesGrabberFactory
{

public:

	/**
	Create and return pointer to FramesGrabber object using camera 
	@param cam_id			camera ID in the system
	@return					smart OpenCV pointer to initialized FramesGrabberCV object
	*/
	static Ptr<FramesGrabber> getFramesGrabber(int cam_id)
	{
		return new FramesGrabberCV(cam_id);
	}

	/**
	Create and return pointer to FramesGrabber object using video or image sequence
	@param filename			videoname, the first image in sequence name or folder path with wildcard
	@return					smart OpenCV pointer to initialized FramesGrabber object
	*/
	static Ptr<FramesGrabber> getFramesGrabber(const String& filename)
	{
		//search for wildcard in filename
		if (filename.find('*') != string::npos)
		{
			return new FramesGrabberSeq(filename);
		}

		return new FramesGrabberCV(filename);
	}

};
