#pragma once

//base class for own implementations of Feature2D algorithms
class MyFeature2D
{
protected:
	cv::Mat mMask;

	virtual int adjustSpecialized(int maxFeatures, const Mat& image) = 0;

public:
	enum 
	{
		Grid, AGAST, sAGAST, BRISK, FAST, sFAST, GFTT, Harris, ORB, SURF, uSURF, SIFT
	};

	MyFeature2D() : mMask(){}
	virtual ~MyFeature2D() {}

	void adjustTo(int maxFeatures, const Mat& image)
	{
		cout << "Adjusting " << getName() << " detector settings to achieve " << maxFeatures << " features in the image...\n";
		int featuresAfterAdjust = adjustSpecialized(maxFeatures, image);

		if(featuresAfterAdjust == -1)
			cout << "Cannot adjust to given number of features!";
		else
			cout << "Features achieved: " << featuresAfterAdjust;
		cout << endl;
	}

	virtual cv::String getName() const = 0;
	virtual bool hasDescriptor() const = 0;

	virtual void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints) const = 0;
	virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const = 0;
	virtual void detectAndCompute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) const = 0;
	virtual int defaultNorm() const = 0;

	static Mat createMask(const Mat& image, float border)
	{
		if (border < FLT_EPSILON) return Mat();

		Mat mask = Mat::zeros(image.size(), image.type());
		Point topLeftCorner(Point(image.size()) * border);
		Point bottomRightCorner(Point(image.size()) * (1.0f - border));
		rectangle(mask, topLeftCorner, bottomRightCorner, Scalar(255), CV_FILLED);

		return mask;
	}

	void setMask(const Mat& mask)
	{
		mask.copyTo(mMask);
	}
	
};

