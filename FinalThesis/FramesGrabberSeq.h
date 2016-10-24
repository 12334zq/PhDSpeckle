#pragma once
#include "FramesGrabber.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

class FramesGrabberSeq : public FramesGrabber
{
	vector<String> mSequence;
	int mSequenceLength;

public:

	/**
	Open given image sequence
	@param pattern		path pattern of sequence using * as wildcard
	*/
	FramesGrabberSeq(const String& pattern) : FramesGrabber()
	{
		//get directory name from path
		size_t pos = pattern.find_last_of('\\');
		if (pos == string::npos) pos = pattern.find_last_of('/');
		if (pos == string::npos) CV_Error(CV_StsBadArg, "Directory not found\n");
		string temp_str(pattern);
		temp_str.resize(pos);
		cout << "Processing images from: " << temp_str << endl;

		//get list of filenames in directory
		vector<String> mFilesList;
		glob(temp_str, mFilesList);
		if (mFilesList.empty()) CV_Error(CV_StsBadArg, "No images in folder\n");

		//find first image of valid format in directory
		size_t numOfFiles = mFilesList.size();
		vector<String> validFormats = { ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff" };
		int formatNum = -1;
		for (int i = 0; i < numOfFiles; i++)
		{
			for (int j = 0; j < validFormats.size(); j++)
			{
				if (mFilesList[i].find(validFormats[j]) != string::npos)
				{
					formatNum = j;
					break;
				}
			}
			if (formatNum != -1) break;
			cout << numOfFiles << "," << formatNum << endl;

			if (i == numOfFiles - 1)
				CV_Error(CV_StsBadArg, "No valid images in folder\n");
		}

		//filter only images of one found valid format from directory
		String formatName = validFormats[formatNum];
		vector<String> unsorted;
		for (int i = 0; i < numOfFiles; i++)
		{
			if (mFilesList[i].find(formatName) != string::npos)
			{
				unsorted.push_back(mFilesList[i]);
			}
		}

		mSequenceLength = static_cast<int>(unsorted.size());

		//sort images according to numbers
		vector<int> idxs;
		for (int i = 0; i < mSequenceLength; i++)
		{
			//get img number from filename
			int sample_number = 0;
			int order = 1;
			pos = mFilesList[i].find(formatName) - 1;
			char check = unsorted[i][pos];

			while (check >= '0' && check <= '9')
			{
				sample_number += static_cast<int>(check - '0') * order;
				order *= 10;
				check = unsorted[i][--pos];
			}

			//check if image isn't already in sequence
			if (find(idxs.begin(), idxs.end(), sample_number) != idxs.end())
				CV_Error(CV_StsBadArg, "Two images with the same number. Cannot state order");

			idxs.push_back(sample_number);
		}
	
		vector<int> sorted_idxs;
		cv::sortIdx(idxs, sorted_idxs, SORT_EVERY_ROW | SORT_ASCENDING);

		mSequence = vector<String>(mSequenceLength);
		for (int i = 0; i < mSequenceLength; i++)
		{
			mSequence[i] = unsorted[sorted_idxs[i]];
		}
	}

	/**
	Release sequence from memory
	*/
	~FramesGrabberSeq()
	{
		mSequence.clear();
	}

	/**
	Open given videofile or images sequence
	@param frame		reference to container for next frame
	@return				acquisition status (false - acquisition stopped / true - acquisition in progress)
	*/
	bool acquire(Mat& output) override
	{
		if (mFrameNumber < mSequenceLength)
		{
			output = imread(mSequence[mFrameNumber++], IMREAD_GRAYSCALE);
			if (mFrameNumber == 1)
			{
				setSize(output.size());
				setType(output.type());
				return true;
			}

			if (output.size() == mSize && output.type() == mType)
				return true;
			if (output.size() != mSize)
				CV_Error(Error::StsUnmatchedSizes, "Consecutive images must have the same size");
			if (output.type() != mType)
				CV_Error(Error::StsUnmatchedFormats, "Consecutive images must have the same data type");
		}
		
		cout << "End of sequence. Press any key\n";
		return false;
	}
};
