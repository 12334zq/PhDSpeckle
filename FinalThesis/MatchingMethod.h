/*
Inherit from Method.h
Basis class for all template matching methods
*/
#pragma once
#include "Method.h"
#include "MetricsFactory.h"
#include "SubPixelEstimatorsFactory.h"

using namespace std;

/**
Base class for area-based template matching methods
*/
class MatchingMethod : public Method
{
private:
	double mTemplRatio; /**< Ratio of template to image size */
	double mMaxShift; /**< Max. shift between consecutive frames */

	/**
	Check settings and initialize matching method basing on the first frame from source 
	@param first			first frame form the source
	*/
	void initMatching(const Mat& first)
	{
		int width = first.cols;
		int height = first.rows;

		//check if image large enough
		CV_Assert(width > 63);
		CV_Assert(height > 63);

		checkMaxShift(width, height);
		checkTemplateRatio();

		//calculate ROI for template
		double borderRatio = (1.0 - mTemplRatio) * 0.5;
		Point tl = Point(round(width * borderRatio), round(height * borderRatio));
		Point br = Point(round(width * (1.0 - borderRatio)), round(height * (1.0 - borderRatio)));
		mTemplateROI = Rect(tl, br);
		cout << "Template area set to: " << mTemplateROI << endl;

		//calculate ROI for search area
		int margin = mSubPixelEstimator->getMargin();
		borderRatio = borderRatio - mMaxShift;
		tl = Point(round(width * borderRatio), round(height * borderRatio));
		if (tl.x < margin) tl.x = margin;
		if (tl.y < margin) tl.y = margin;
		br = Point(round(width * (1.0 - borderRatio)), round(height * (1.0 - borderRatio)));
		if (br.x > width - margin) br.x = width - margin;
		if (br.y > height - margin) br.y = height - margin;
		mSearchROI = Rect(tl, br);
		cout << "Search area set to: " << mSearchROI << endl;

		//maximum translation of template
		mMaxTranslation = mTemplateROI.tl() - mSearchROI.tl();

		//initialize template
		first(mTemplateROI).copyTo(mTemplate);
	}

	/**
	Check if maximum shift value is in correct range 
	@param width			width of the image
	@param height			height of the image
	*/
	void checkMaxShift(int width, int height)
	{
		//assure that maxShift is sufficient to compute sub-pixel estimation
		double min = cv::min(mMaxShift*width, mMaxShift*height);
		CV_Assert(min > mSubPixelEstimator->getMargin() + 1);

		//assure that maxShift is not higher than limit
		double maxShiftLimit = 0.48;
		if (mMaxShift > maxShiftLimit)
		{
			cout << "Max. shift automatically limited to " << maxShiftLimit * 100 << "%\n";
			mMaxShift = maxShiftLimit;
		}
	}

	/**
	Check if template ratio value is in correct range
	*/
	void checkTemplateRatio()
	{
		//check ratio and optionally limit to minimum/maximum 
		double minRatio = 0.01, maxRatio = 1.0 - 2 * mMaxShift;
		if (mTemplRatio < minRatio)
		{
			cout << "Too small template. Automatically increased to " << minRatio * 100 << "%\n";
			mTemplRatio = minRatio;
		}
		else if (mTemplRatio > maxRatio)
		{
			cout << "Too large template. Automatically limited to " << maxRatio * 100 << "%\n";
			mTemplRatio = maxRatio;
		}
	}
	
protected:
	Ptr<Metric> mMetric; /**< Object of metric to use */
	Ptr<SubPixelEstimator> mSubPixelEstimator; /**< Object for sub-pixel accuracy estimation */
	Rect mTemplateROI; /**< ROI for template */
	Rect mSearchROI; /**< ROI for search */
	Point mMaxTranslation; /**< Max. translation */
	Mat mTemplate; /**< Previous template image */

public:
	explicit MatchingMethod(const String& name, const Mat& first, int metric, double tempRatio, double maxShift) : Method(name), mTemplRatio(tempRatio), mMaxShift(maxShift)
	{
		mSubPixelEstimator = SubPixelEstimatorsFactory::getEstimator(SubPixelEstimator::GAUSS5);
		initMatching(first);
		mMetric = MetricsFactory::getMetric(metric);
		mMetric->reloadCache(mTemplate);
		addToName("_" + mMetric->getName());
	}

	~MatchingMethod(){}

};
