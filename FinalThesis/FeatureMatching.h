#pragma once
#include "FeaturesMethod.h"

/**
Feature tracking with descriptors matching using FLANN and Brute force algorithms
*/
class FeatureMatching : public FeaturesMethod
{
	Ptr<DescriptorMatcher> mMatcher; /**< Pointer to descriptor matcher object */
	String mMatcherName; /**< Matcher name */
	Mat mPrevDescriptors; /**< Previously calculated descriptors */
	vector<KeyPoint> mNewKeypoints; /**< Memory allocated for new keypoints */
	Mat mNewDescriptors; /**< Memory allocated for new descriptors */
	int mNumBestMatches; /**< Maximum number of best matches to leave */
	int mSumFeatures = 0; /**< Sum features in next iterations (debug) */

	/**
	Find rigid transformation matrix for the next frame
	@param frame		next frame
	@return				transformation matrix
	*/
	Mat getTransform(const Mat& img) override
	{
		Mat M(2, 3, CV_64F);

		//compute descriptors for image
		mDetector->detectAndCompute(img, mDetectorMask, mNewKeypoints, mNewDescriptors);

		//convert to type valid for FLANN
		if (mMatcherName == "FlannBased" && mNewDescriptors.type() != CV_32F) {
			mNewDescriptors.convertTo(mNewDescriptors, CV_32F);
		}

		//match descriptors to previous image
		vector<DMatch> matches;
		mMatcher->match(mPrevDescriptors, mNewDescriptors, matches);

		//stop if too low number of matches to achieve reasonable result
		if (matches.size() < 9)
		{
			//copy keypoints and descriptors
			mPrevKeypoints = mNewKeypoints;
			mNewDescriptors.copyTo(mPrevDescriptors);
			return M;
		}

		//select the best matches only
		Mat index;
		int nbMatch = int(matches.size());
		Mat tab(nbMatch, 1, CV_32F);
		for (int i = 0; i<nbMatch; i++)
		{
			tab.at<float>(i, 0) = matches[i].distance;
		}
		sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);

		const int bestMatches = min(nbMatch / 2, mNumBestMatches);
		vector<DMatch> good_matches(bestMatches);
		for (int i = 0; i < bestMatches; i++)
		{
			good_matches[i] = matches[index.at<int>(i, 0)];
		}

		////draw results if needed
		//if (mDrawResult)
		//{
		//	drawMatches(mPrevFrame, mPrevKeypoints, img, mNewKeypoints,
		//		good_matches, mResultImg, Scalar::all(-1), Scalar::all(-1),
		//		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//}

		//calculate transformation matrix
		vector<Point2f> pA(bestMatches), pB(bestMatches);
		for (int i = 0; i < bestMatches; i++)
		{
			pA[i] = mPrevKeypoints[good_matches[i].queryIdx].pt;
			pB[i] = mNewKeypoints[good_matches[i].trainIdx].pt;
		}

		if (mEstimationType == 1)
		{
			bool result = RANSAC(pA, pB, 0.5);
			if (!result) cout << "RANSAC failed!" << endl;
		}

		if (mDrawResult)
		{
			img.copyTo(mResultImg);
			cvtColor(mResultImg, mResultImg, CV_GRAY2BGR);

			for (int i = 0; i < pA.size(); i++)
			{
				circle(mResultImg, pA[i], mResultImg.cols / 80, Scalar(0, 0, 255), 2);
			}
			for (int i = 0; i < pB.size(); i++)
			{
				arrowedLine(mResultImg, pA[i], pB[i], Scalar(0, 255, 0));
				circle(mResultImg, pB[i], mResultImg.cols / 80, Scalar(0, 255, 0), 2);
			}
		}

		getRTMatrix(pA, pB, pA.size(), M);

		//copy keypoints and descriptors
		mPrevKeypoints = mNewKeypoints;
		mNewDescriptors.copyTo(mPrevDescriptors);

		return M;
	}

public:
	FeatureMatching(const Mat& first, const String& detector, const String& matcher, int maxFeatures, int estimation) : FeaturesMethod("FeatureMatching", first, detector, maxFeatures, estimation)
	{
		CV_Assert(mDetector->descriptorSize() > 0);

		if (matcher == "FlannBased")
		{
			mMatcher = new FlannBasedMatcher();
			mMatcherName = matcher;
		}
		else
		{
			mMatcher = new BFMatcher(mDetector->defaultNorm());
			mMatcherName = "BruteForce";
		}
		addToName("_" + mMatcherName);

		mDetector->compute(first, mPrevKeypoints, mPrevDescriptors);
		if (mMatcherName == "FlannBased" && mPrevDescriptors.type() != CV_32F) {
			mPrevDescriptors.convertTo(mPrevDescriptors, CV_32F);
		}

		mNumBestMatches = 50;
	}

	/**
	Get displacement with sub-pixel accuracy using descriptors matching
	@param frame		next frame
	@return				displacement with respect to previous frame
	*/
	Point3f getDisplacement(const Mat& img) override
	{
		mSumFeatures += mPrevKeypoints.size();
		Mat transform = getTransform(img);

		img.copyTo(mPrevFrame);

		return Point3f(transform.at<double>(0, 2), transform.at<double>(1, 2), asin(transform.at<double>(1, 0)) * 180 / CV_PI);
	}

	/**
	Get number of accumulated features (debug)
	*/
	int getFeatures() override
	{
		return mSumFeatures;
	}
};
