#pragma once
#include "FramesGrabberFactory.h"
#include "MethodsFactory.h"

using namespace std;
using namespace cv;

/**
Laser Speckle Velocimeter
Main interface of the system.
*/
class LaserSpeckleVelocimeter
{
	Ptr<FramesGrabber> mFramesGrabber; /**< Pointer to frames grabber object */
	Ptr<Method> mMethod; /**< Pointer to method object */
	Point3f mDisplacement{ 0 }; /**< Total displacement counter */
	Point2f mVelocity{ 0 }; /**< Velocity of last displacement  */
	double mTime = 0; /**< Calculation time of last displacement */
	uint64 mFrameNumber = 0; /**< Actual frame number */
	double PX2MM; /**< Calibration factor (unused yet) */

public:
	/**
	Explicit constructor
	@param path			path to videofile / the first image in the sequence / folder path with wildcard '*' / default camera if "0" 
	@param method		method number according to Method::METHODS enum
	@param params		method parameters in MethodParams structure
	@param px2mm		calibration factor (unused yet)
	@param draw			results drawing flag
	*/
	explicit LaserSpeckleVelocimeter(const String& path, int method, MethodParams params, double px2mm = 1.0, bool draw = false) : PX2MM(px2mm)
	{
		if (path == "0") 
			mFramesGrabber = FramesGrabberFactory::getFramesGrabber(0);
		else
			mFramesGrabber = FramesGrabberFactory::getFramesGrabber(path);

		Mat first;
		CV_Assert(mFramesGrabber->acquire(first));

		mMethod = MethodsFactory::getMethod(method, first, params);

		//display results for debugging
		cout << "Method: " << mMethod->getName() << endl;
		if (draw) mMethod->drawingOnOff();
	}

	/**
	Get measured displacement
	@return				measured displacement
	*/
	Point3f getDisplacement() const { return mDisplacement; }

	/**
	Get method name
	@return				mehod name
	*/
	String getMehodName() const { return mMethod->getName(); }

	/**
	Get measured velocity
	@return				measured velocity
	*/
	Point2f getVelocity() const { return mVelocity; }

	/**
	Get result image
	@return				Mat with result
	*/
	Mat getResultImg() const { return mMethod->getResultImg(); }

	/**
	Get processing time for last frame
	@return				processing time in seconds
	*/
	double getTime() const { return mTime; }

	/**
	Get actual frame number
	@return				frame number
	*/
	uint64 getFrameNumber() const { return mFrameNumber; }

	int getFeatures() const { return mMethod->getFeatures(); }

	/**
	Main method of the project. It activates complete process of measuring translation with given method in the next frame.
	The actual frame can be accessed by argument (otherwise put noArray()) 
	It returns boolean status of acquisition as a control output for user interface
	@param output		actual frame
	@return				status of acquisition, true - if acquisition in progress / false - if acquisition has ended
	*/
	bool nextMeasurement(OutputArray output)
	{
		Mat frame;
		if (!mFramesGrabber->acquire(frame)) return false;
		mFrameNumber++;

		//start timer
		mTime = static_cast<double>(getTickCount());

		//measure next displacement
		Point3f measurement = mMethod->getDisplacement(frame);

		//increase total displacement
		mDisplacement = Point3f(measurement.x * PX2MM, measurement.y * PX2MM, measurement.z);

		//stop timer
		mTime = (static_cast<double>(getTickCount()) - mTime) / getTickFrequency();

		//calculate velocity
		double FPS = mFramesGrabber->getFPS();
		if (FPS < 0) FPS = 1.0 / mTime;
		mVelocity.x = sqrt(measurement.x * measurement.x + measurement.y * measurement.y) * FPS * PX2MM;
		mVelocity.y = abs(measurement.z) * FPS;

		if (output.needed()) frame.copyTo(output);

		return true;
	}
};
