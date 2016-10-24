#pragma once
#include <opencv2/core/cvstd.hpp>
#include <fstream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>

using namespace cv;
using namespace std;

class Evaluator
{
	vector<Point3f> mGroundTruth;
	vector<Point3f> mResults;
	int mGroundTruthSize;
	int mScale;
	Point3f mAvgError;
	Point3f mLastError;
	bool mStatus = false;

public:
	Evaluator() : mGroundTruth(), mResults(0), mGroundTruthSize(0), mScale(0), mAvgError(0), mLastError(0){}

	Evaluator(const String& path) : mResults(0), mGroundTruthSize(0), mAvgError(0), mLastError(0)
	{
		//get directory name from path
		size_t pos = path.find_last_of('\\');
		if (pos == string::npos) pos = path.find_last_of('/');
		if (pos == string::npos) CV_Error(CV_StsBadArg, "Directory not found\n");
		string groundPath(path);
		groundPath.resize(pos);
		groundPath += "//moves.txt";

		//read ground truth file to vector
		ifstream groundFile(groundPath);

		if (!groundFile.is_open())
			cout << "Failed to open " << groundPath << ". Evaluation won't work" << endl;
		else
		{
			string line;
			bool first = true;
			while (getline(groundFile, line))
			{
				if (first)
				{
					mScale = stod(line);
					first = false;
				}
				else
				{
					pos = line.find(',');
					string x_string(line);
					x_string.resize(pos);
					double x_val = stod(x_string);
					line.erase(0, pos + 1);

					pos = line.find(',');
					string y_string(line);
					if (pos != string::npos)
					{
						y_string.resize(pos);
						line.erase(0, pos + 1);
					}
					else
						line.clear();
					double y_val = stod(y_string);

					double z_val = 0;
					if (!line.empty())z_val = stod(line);

					//y axis is inverted in OpenCV
					mGroundTruth.push_back(Point3f(x_val, y_val, z_val));
					mGroundTruthSize++;
					mStatus = true;
				}
			}
		}

		groundFile.close();
	}

	~Evaluator()
	{
		mGroundTruth.clear();
	}

	void evaluate(Point3f result, int frameNum)
	{
		if (mGroundTruthSize == 0 || frameNum > mGroundTruthSize || frameNum < 1) return;

		Point3f correction(0);
		if (frameNum > 1) correction = mGroundTruth[frameNum - 2];

		Point3f invResult = Point3f(-result.x, result.y, -result.z);
		mResults.push_back(invResult);
		float x_err = abs(invResult.x - (mGroundTruth[frameNum - 1].x - correction.x));
		float y_err = abs(invResult.y - (mGroundTruth[frameNum - 1].y - correction.y));
		float deg_err = abs(invResult.z - (mGroundTruth[frameNum - 1].z - correction.z));
		mLastError = Point3f(x_err, y_err, deg_err);
		mAvgError += mLastError;
	}

	bool getStatus() const { return mStatus; }

	Mat getPathImg() const
	{
		vector<Point2f> GroundPath, ResultPath;
		for (int i = 0; i < mGroundTruth.size(); i++)
		{
			GroundPath.push_back(Point2f(mGroundTruth[i].x, mGroundTruth[i].y));
		}
		for (int i = 0; i < mResults.size(); i++)
		{
			ResultPath.push_back(Point2f(mResults[i].x, mResults[i].y));
		}

		Rect rect = boundingRect(GroundPath) | boundingRect(ResultPath);
		Point2f tl = rect.tl();
		Mat image = Mat::zeros(rect.size(), CV_8UC3);
		resize(image, image, Size(), mScale, mScale, INTER_AREA);

		for (int i = 0; i < GroundPath.size() - 1; i++)
		{
			//drawMarker(image, mScale * (mGroundTruth[i] - tl),  Scalar(0, 0, 255), MARKER_DIAMOND, 5);
			line(image, mScale * (GroundPath[i] - tl), mScale * (GroundPath[i + 1] - tl), Scalar(0, 0, 255), mScale * 15);
		}
		for (int i = 0; i < ResultPath.size() - 1; i++)
		{
			//drawMarker(image, mScale * (mResults[i] - tl), Scalar(0, 255, 0), MARKER_DIAMOND, 5);
			line(image, mScale * (ResultPath[i] - tl), mScale * (ResultPath[i + 1] - tl), Scalar(0, 255, 0), mScale * 5);
		}

		return image;
	}

	Point3f getLastError() const
	{
		return mLastError;
	}

	Point3f getAvgError() const
	{
		return mAvgError / mGroundTruthSize;
	}

	double getMeanMotion() const
	{
		double mean = 0;
		for (int i = 1; i < mGroundTruthSize; i++)
		{
			Point3f p = mGroundTruth[i] - mGroundTruth[i - 1];
			mean += sqrt(p.x*p.x + p.y*p.y);
		}
		mean /= static_cast<double>(mGroundTruthSize - 1);

		return mean;
	}

};
