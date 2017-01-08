#pragma once
#include "RegularGrid.h"
#include "MyAGAST.h"
#include "MySAGAST.h"
#include "MyBRISK.h"
#include "MyFAST.h"
#include "MySFAST.h"
#include "MyGFTT.h"
#include "MyHarris.h"
#include "MyORB.h"
#include "MySURF.h"
#include "MyUSURF.h"
#include "MySIFT.h"
#include "MyStarDetector.h"

class MyFeature2DFactory
{
public:

	static cv::Ptr<MyFeature2D> create(int detector, int maxFeatures, const Mat& image, const Mat& mask)
	{
		Ptr<MyFeature2D> result;

		switch(detector)
		{
		case MyFeature2D::Grid: { result = makePtr<RegularGrid>(); break; }
		case MyFeature2D::AGAST: { result = makePtr<MyAGAST>(); break; }
		case MyFeature2D::sAGAST: { result = makePtr<MySAGAST>(); break; }
		case MyFeature2D::BRISK: { result = makePtr<MyBRISK>(); break; }
		case MyFeature2D::FAST: { result = makePtr<MyFAST>(); break; }
		case MyFeature2D::sFAST: { result = makePtr<MySFAST>(); break; }
		case MyFeature2D::GFTT: { result = makePtr<MyGFTT>(); break; }
		case MyFeature2D::Harris: { result = makePtr<MyHarris>(); break; }
		case MyFeature2D::ORB: { result = makePtr<MyORB>(); break; }
		case MyFeature2D::SURF: { result = makePtr<MySURF>(); break; }
		case MyFeature2D::uSURF: { result = makePtr<MyUSURF>(); break; }
		case MyFeature2D::SIFT: { result = makePtr<MySIFT>(); break; }
		case MyFeature2D::Star: { result = makePtr<MyStarDetector>(); break; }
		default: {result = makePtr<MyFAST>(); }
		}

		result->setMask(mask);
		result->adjustTo(maxFeatures, image);

		return result;
	}
};

