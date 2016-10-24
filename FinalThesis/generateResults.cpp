/**
LSVelocimeter project
@author Krzysztof Kotowski
*/
#include "LaserSpeckleVelocimeter.h"
#include "Evaluator.h"

int genres(int argc, char** argv)
{
	cout << "Speckle velocimetry - Krzysztof Kotowski\n";

	int method = Method::FULL_FFT;
	bool draw = false;
	double px2mm = 1.0;
	String px_mm = "px";

	MethodParams params;
	params.metric = Metric::NXC;
	params.templRatio = 0.8;
	params.maxShift = 0.1;
	params.layers = 3;
	params.maxFeatures = 200;
	params.detector = "FAST";
	params.RANSAC = 1;
	params.matcher = "BruteForce";

	String mainPath = "C:\\Users\\Krzysztof\\Downloads\\data";
	if (argc > 1) mainPath = argv[1];

	ofstream file;
	int idx;
	vector<String> stringVec1, stringVec2;
	vector<String>::iterator stringIter1, stringIter2;
	vector<int> intVec1, intVec2;
	vector<int>::iterator intIter1, intIter2;
	vector<double> doubleVec1, doubleVec2;
	vector<double>::iterator doubleIter1, doubleIter2;

	String path = mainPath + "\\gen10_2\\*.png";

	/*
	Different metrics for full search spatial and FFT
	*/
	/*ofstream metrics("metrics10gauss3.csv");
	metrics << "Metric;FFT;FFT_Acc;LRP;LRP_Acc;Spiral;Spiral_Acc\n";

	for (int i = 0; i < Metric::MAD; i++)
	{
	params.metric = i;
	metrics << i << ";";
	for (int j = 0; j <= Method::SPIRAL; j++)
	{
	if (j == Method::FULL_SPATIAL) continue;
	if ((i == Metric::SAD || i == Metric::MAD) && j == 0)
	{
	metrics << 0 << ";" << 0 << ";";
	continue;
	}

	LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	Evaluator evaluator(path);

	double sumTime = 0;
	uint64 frameNumber = 0;

	for (;;)
	{
	Mat frame;
	if (!LSV.nextMeasurement(frame)) break;

	auto V = LSV.getVelocity();
	auto displacement = LSV.getDisplacement();
	frameNumber = LSV.getFrameNumber();
	auto time = LSV.getTime();

	accumulate results
	if (frameNumber > 1)
	{
	sumTime += time;
	}

	evaluator.evaluate(displacement, frameNumber);
	}

	sumTime /= (frameNumber - 1.0);

	auto r = evaluator.getAvgError();
	metrics << 1000 * sumTime << ";";
	metrics << sqrt(r.x*r.x + r.y*r.y) << ";";
	}

	metrics << endl;
	}
	metrics.close();*/

	/*
	Different template sizes for metrics
	*/
	//params.metric = Metric::NXC;
	//params.maxShift = 0.1;
	//ofstream templates("templates10.csv");
	//templates << "TemplRatio;SSD;SSD_Acc;NSSD;NSSD_Acc;XC;XC_Acc;NXC;NXC_Acc;ZXC;ZXC_Acc;ZNXC;ZNXC_Acc;SAD;SAD_acc;\n";
	//for (auto i = 0.01; i < 0.801; i += 0.01)
	//{
	//	params.templRatio = i;
	//	templates << i << ";";
	//	for (int j = 0; j <= Metric::SAD; j++)
	//	{
	//		params.metric = j;
	//		method = Method::FULL_FFT;
	//		if (j == Metric::SAD) method = Method::FULL_SPATIAL;

	//		LaserSpeckleVelocimeter LSV(path, method, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);

	//			if (waitKey(1) != -1) break;
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		templates << 1000 * sumTime << ";";
	//		templates << sqrt(r.x*r.x + r.y*r.y) << ";";
	//	}
	//	templates << endl;
	//}
	//templates.close();

	/*
	Different max shifts
	*/
	//params.templRatio = 0.5;
	//params.metric = Metric::NXC;
	//file.open("shifts10.csv");
	//file << "Shift;FFT;FFT_Acc;LRP;LRP_Acc;Spiral;Spiral_Acc\n";
	//for (auto i = 0.05; i < 0.26; i += 0.01)
	//{
	//	params.maxShift = i;
	//	file << i << ";";
	//	for (int j = 0; j <= Method::SPIRAL; j++)
	//	{
	//		if (j == Method::FULL_SPATIAL) continue;

	//		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);

	//			if (waitKey(1) != -1) break;
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		file << 1000 * sumTime << ";";
	//		file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//	}
	//	file << endl;
	//}
	//file.close();

	///*
	//Different template sizes for SAD
	//*/
	//params.metric = Metric::SAD;
	//ofstream templatesSAD("templatesSAD.csv");
	//templatesSAD << "TemplRatio;LRP;LRP_Acc;Spiral;Acc\n";
	//for (auto i = 0.4; i < 0.801; i += 0.01)
	//{
	//	params.templRatio = i;
	//	templatesSAD << i << ";";
	//	for (int j = Method::LRP; j <= Method::SPIRAL; j++)
	//	{
	//		if (j == Method::FULL_SPATIAL || j == Method::FULL_FFT) continue;

	//		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);

	//			if (waitKey(1) != -1) break;
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		templatesSAD << 1000 * sumTime << ";";
	//		templatesSAD << sqrt(r.x*r.x + r.y*r.y) << ";";
	//	}
	//	templatesSAD << endl;
	//}
	//templatesSAD.close();

	/*
	Vibrations influence
	*/
	//params.metric = Metric::NXC;
	//file.open("vibrations10.csv");
	//file << "Var;FFT;Acc;LRP;Acc;Spiral;Acc;Flow_sFAST_480;Flow_Acc1;Flow_sAGAST150;Flow2_Acc;Flow_GFTT210;Flow3_Acc;BRISK;BRISK_Acc\n";
	//stringVec1 = { "\\vib0-2\\*.png", "\\vib0-5\\*.png", "\\vib1\\*.png", "\\vib3\\*.png", "\\vib5\\*.png", "\\vib6\\*.png",
	//	"\\vib7\\*.png", "\\vib10\\*.png", "\\vib15\\*.png", "\\vib17\\*.png", "\\vib18\\*.png", "\\vib20\\*.png" };
	//intVec1 = { 0, 0, 1, 3, 5, 6, 7, 10, 15, 17, 18, 20 };

	//stringVec2 = { "FAST", "sAGAST", "Harris" };
	//intVec2 = { 400, 100, 150 };
	//params.maxFeatures = intVec2[0];
	//params.detector = stringVec2[0];
	//idx = 1;

	//for (stringIter1 = stringVec1.begin(), intIter1 = intVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++intIter1)
	//{
	//	path = mainPath + *stringIter1;

	//	for (int j = 0; j <= Method::FEATURE_MATCHING; j++)
	//	{
	//		if (j == Method::FULL_SPATIAL) continue;
	//		if (j == Method::FEATURE_MATCHING)
	//		{
	//			params.maxFeatures = 250;
	//			params.detector = "BRISK";
	//		}

	//		params.layers = 3;
	//		if (j == Method::OPTICAL_FLOW) params.layers = 4;

	//		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		if (j == 0) file << evaluator.getMeanMotion() << ";";
	//		file << 1000 * sumTime << ";";
	//		file << sqrt(r.x*r.x + r.y*r.y) << ";";

	//		if (j == Method::OPTICAL_FLOW && idx < stringVec2.size())
	//		{
	//			params.maxFeatures = intVec2[idx];
	//			params.detector = stringVec2[idx++];
	//			j--;
	//		}
	//		else
	//		{
	//			params.maxFeatures = intVec2[0];
	//			params.detector = stringVec2[0];
	//		}
	//	}

	//	idx = 1;
	//	file << endl;
	//}
	//file.close();

	/*
	Constvec influence
	*/
	//params.metric = Metric::NXC;
	//file.open("constvec10.csv");
	//file << "Var;FFT;Acc;LRP;Acc;Spiral;Acc;Flow_sFAST_480;Flow_Acc1;Flow_sAGAST150;Flow2_Acc;Flow_Harris150;Flow3_Acc;BRISK;BRISK_Acc\n";
	//stringVec1 = { "\\const1_5\\*.png", "\\const2_5\\*.png", "\\const3_2\\*.png", "\\const5_8\\*.png", 
	//	"\\const9_0\\*.png", "\\const9_1\\*.png", "\\const9_2\\*.png", "\\const9_3\\*.png", "\\const9_4\\*.png", "\\const9_5\\*.png",
	//	"\\const9_6\\*.png", "\\const9_7\\*.png", "\\const9_8\\*.png", "\\const9_9\\*.png", "\\const10_0\\*.png",
	//	"\\const14_4\\*.png", "\\const16_6\\*.png", "\\const18_1\\*.png", "\\const19_9\\*.png", };
	//doubleVec1 = { 1.5, 2.5, 3.2, 5.8, 9.0, 9.1 ,9.2,9.3,9.4,9.5,9.6,9.7,9.8, 9.9, 10.0, 14.4, 16.6, 18.1, 19.9 };

	//stringVec2 = { "FAST", "sAGAST", "Harris" };
	//intVec2 = { 400, 100, 150 };
	//params.maxFeatures = intVec2[0];
	//params.detector = stringVec2[0];
	//idx = 1;

	//for (stringIter1 = stringVec1.begin(), doubleIter1 = doubleVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++doubleIter1)
	//{
	//	file << *doubleIter1 << ";";
	//	path = mainPath + *stringIter1;

	//	for (int j = 0; j <= Method::FEATURE_MATCHING; j++)
	//	{
	//		if (j == Method::FULL_SPATIAL) continue;
	//		if (j == Method::FEATURE_MATCHING)
	//		{
	//			params.maxFeatures = 250;
	//			params.detector = "BRISK";
	//		}

	//		params.layers = 3;
	//		if (j == Method::OPTICAL_FLOW) params.layers = 4;

	//		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		file << 1000 * sumTime << ";";
	//		file << sqrt(r.x*r.x + r.y*r.y) << ";";

	//		if (j == Method::OPTICAL_FLOW && idx < stringVec2.size())
	//		{
	//			params.maxFeatures = intVec2[idx];
	//			params.detector = stringVec2[idx++];
	//			j--;
	//		}
	//		else
	//		{
	//			params.maxFeatures = intVec2[0];
	//			params.detector = stringVec2[0];
	//		}
	//	}

	//	idx = 1;
	//	file << endl;
	//}
	//file.close();

	/*
	Const vec metrics
	*/
	//file.open("lockingMetrics10.csv");
	//params.layers = 3;
	//file << "Length;SSD;NSSD;XC;NXC;ZXC;ZNXC;SAD\n";
	//stringVec1 = { "\\const1_5\\*.png", "\\const2_5\\*.png", "\\const3_2\\*.png", "\\const5_8\\*.png",
	//	"\\const9_0\\*.png", "\\const9_1\\*.png", "\\const9_2\\*.png", "\\const9_3\\*.png", "\\const9_4\\*.png", "\\const9_5\\*.png",
	//	"\\const9_6\\*.png", "\\const9_7\\*.png", "\\const9_8\\*.png", "\\const9_9\\*.png", "\\const10_0\\*.png",
	//	"\\const14_4\\*.png", "\\const16_6\\*.png", "\\const18_1\\*.png", "\\const19_9\\*.png", };
	//doubleVec1 = { 1.5, 2.5, 3.2, 5.8, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 14.4, 16.6, 18.1, 19.9 };

	//for (stringIter1 = stringVec1.begin(), doubleIter1 = doubleVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++doubleIter1)
	//{
	//	file << *doubleIter1 << ";";
	//	path = mainPath + *stringIter1;

	//	for (int j = 0; j <= Metric::SAD; j++)
	//	{
	//		method = Method::FULL_FFT;
	//		if (j == Metric::SAD) method = Method::LRP;
	//		params.metric = j;
	//		LaserSpeckleVelocimeter LSV(path, method, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//	}
	//	file << endl;
	//}
	//file.close();


	///////*
	//////Direction vec influence
	//////*/
	//////params.metric = Metric::NXC;
	//////file.open("directions5.csv");
	//////file << "Var;FFT;Acc;LRP;Acc;Spiral;Acc;Flow_sFAST_480;Flow_Acc1;Flow_sAGAST150;Flow2_Acc;Flow_GFTT210;Flow3_Acc\n";
	//////stringVec1 = { "\\gen14_0_0\\*.png", "\\gen10_0_45\\*.png", "\\gen14_0_90\\*.png", "\\gen10_0_135\\*.png", "\\gen14_0_180\\*.png", "\\gen10_0_225\\*.png" };
	//////intVec1 = { 0, 45, 90, 135, 180, 225 };

	//////stringVec2 = { "sFAST", "sAGAST", "GFTT" };
	//////intVec2 = { 480, 150, 210 };
	//////params.maxFeatures = intVec2[0];
	//////params.detector = stringVec2[0];
	//////idx = 1;

	//////for (stringIter1 = stringVec1.begin(), intIter1 = intVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++intIter1)
	//////{
	//////	file << *intIter1 << ";";
	//////	path = mainPath + *stringIter1;

	//////	for (int j = 0; j <= Method::OPTICAL_FLOW; j++)
	//////	{
	//////		if (j == Method::FULL_SPATIAL) continue;

	//////		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//////		Evaluator evaluator(path);

	//////		double sumTime = 0;
	//////		uint64 frameNumber = 0;

	//////		for (;;)
	//////		{
	//////			Mat frame;
	//////			if (!LSV.nextMeasurement(frame)) break;

	//////			auto V = LSV.getVelocity();
	//////			auto displacement = LSV.getDisplacement();
	//////			frameNumber = LSV.getFrameNumber();
	//////			auto time = LSV.getTime();

	//////			//accumulate results
	//////			if (frameNumber > 1)
	//////			{
	//////				sumTime += time;
	//////			}

	//////			evaluator.evaluate(displacement, frameNumber);
	//////		}

	//////		sumTime /= (frameNumber - 1.0);

	//////		auto r = evaluator.getAvgError();
	//////		file << 1000 * sumTime << ";";
	//////		file << sqrt(r.x*r.x + r.y*r.y) << ";";

	//////		if (j == Method::OPTICAL_FLOW && idx < stringVec2.size())
	//////		{
	//////			params.maxFeatures = intVec2[idx];
	//////			params.detector = stringVec2[idx++];
	//////			j--;
	//////		}
	//////		else
	//////		{
	//////			params.maxFeatures = intVec2[0];
	//////			params.detector = stringVec2[0];
	//////		}
	//////	}

	//////	idx = 1;
	//////	file << endl;
	//////}
	//////file.close();
	////
	/*
	Noise influence SUMMARY
	*/
	//params.metric = Metric::NXC;
	//file.open("noise10_SURF.csv");
	//file << "Var;FFT;Acc;LRP;Acc;Spiral;Acc;Flow_sFAST_480;Flow_Acc1;Flow_sAGAST150;Flow2_Acc;Flow_Harris150;Flow3_Acc;BRISK;BRISK_Acc\n";
	//stringVec1 = { "\\gen10_2_near\\*.png", "\\gen10_2_n05\\*.png", "\\gen10_2_n1\\*.png", "\\gen10_2_n3\\*.png" ,"\\gen10_2_n5\\*.png", "\\gen10_2_n6\\*.png",
	//	"\\gen10_2_n7\\*.png", "\\gen10_2_n9\\*.png", "\\gen10_2_n10\\*.png", "\\gen10_2_n12\\*.png", "\\gen10_2_n15\\*.png", "\\gen10_2_n20\\*.png" };
	//doubleVec1 = { 0, 0.5, 1, 3, 5, 6, 7, 9, 10, 12, 15, 20 };

	//stringVec2 = { "FAST", "sAGAST", "Harris" };
	//intVec2 = { 400, 100, 150 };
	//params.maxFeatures = intVec2[0];
	//params.detector = stringVec2[0];
	//idx = 1;

	//for (stringIter1 = stringVec1.begin(), doubleIter1 = doubleVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++doubleIter1)
	//{
	//	file << *doubleIter1 << ";";
	//	path = mainPath + *stringIter1;

	//	for (int j = Method::FEATURE_MATCHING; j <= Method::FEATURE_MATCHING; j++)
	//	{
	//		if (j == Method::FULL_SPATIAL) continue;
	//		if (j == Method::FEATURE_MATCHING)
	//		{
	//			params.maxFeatures = 50;
	//			params.detector = "SURF";
	//		}

	//		params.layers = 3;
	//		if (j == Method::OPTICAL_FLOW) params.layers = 4;

	//		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		file << 1000 * sumTime << ";";
	//		file << sqrt(r.x*r.x + r.y*r.y) << ";";

	//		if (j == Method::OPTICAL_FLOW && idx < stringVec2.size())
	//		{
	//			params.maxFeatures = intVec2[idx];
	//			params.detector = stringVec2[idx++];
	//			j--;
	//		}
	//		else
	//		{
	//			params.maxFeatures = intVec2[0];
	//			params.detector = stringVec2[0];
	//		}
	//	}

	//	idx = 1;
	//	file << endl;
	//}
	//file.close();
	//
	///*
	//Noise influence metrics
	//*/
	//file.open("noiseMetrics10.csv");
	//params.layers = 3;
	//file << "StdDev;SSD;NSSD;XC;NXC;ZXC;ZNXC;SAD\n";
	//stringVec1 = { "\\gen10_2_near\\*.png", "\\gen10_2_n05\\*.png", "\\gen10_2_n1\\*.png", "\\gen10_2_n3\\*.png", "\\gen10_2_n5\\*.png", "\\gen10_2_n6\\*.png",
	//	"\\gen10_2_n7\\*.png", "\\gen10_2_n9\\*.png", "\\gen10_2_n10\\*.png", "\\gen10_2_n12\\*.png", "\\gen10_2_n15\\*.png", "\\gen10_2_n20\\*.png" };
	//doubleVec1 = { 0, 0.5, 1, 3, 5, 6, 7, 9, 10, 12, 15, 20 };

	//for (stringIter1 = stringVec1.begin(), doubleIter1 = doubleVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++doubleIter1)
	//{
	//	file << *doubleIter1 << ";";
	//	path = mainPath + *stringIter1;

	//	for (int j = 0; j <= Metric::SAD; j++)
	//	{
	//		method = Method::FULL_FFT;
	//		if (j == Metric::SAD) method = Method::FULL_SPATIAL;
	//		params.metric = j;
	//		LaserSpeckleVelocimeter LSV(path, method, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//	}
	//	file << endl;
	//}
	//file.close();


	///*
	//Optical flow performance TRANSLATION
	//*/
	//path = mainPath + "\\gen10_2\\*.png";
	//file.open("opticalFlow10_extend.csv");
	//stringVec1 = { "Agast", "sAgast", "FAST", "sFAST", "GFTT", "Harris", "U-SURF" };

	//for (stringIter1 = stringVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1)
	//{
	//	params.detector = *stringIter1;
	//	file << "Feat;" << *stringIter1 << "_time;" << *stringIter1 << "_acc;" << endl;

	//	int oldFeat = 0;
	//	for (int f = 10; f < 501; f += 10)
	//	{
	//		params.maxFeatures = f;
	//		LaserSpeckleVelocimeter LSV(path, Method::OPTICAL_FLOW, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		int newFeat = round(LSV.getFeatures() / static_cast<double>(frameNumber));
	//		if (newFeat != oldFeat)
	//		{
	//			oldFeat = newFeat;
	//			file << newFeat << ";";
	//			file << 1000 * sumTime << ";";
	//			file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//			file << endl;
	//		}
	//	}
	//}
	//file.close();

	/*
	Optical flow performance ROTATION
	*/
	//path = mainPath + "\\rot5_0\\*.png";
	//file.open("opticalFlowRot10_extend.csv");
	//stringVec1 = { "Agast", "sAgast", "FAST", "sFAST", "GFTT", "Harris", "U-SURF" };

	//for (stringIter1 = stringVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1)
	//{
	//	params.detector = *stringIter1;
	//	file << "Feat;" << *stringIter1 << "_time;" << *stringIter1 << "_acc;" << *stringIter1 << "_rotAcc;" << endl;

	//	int oldFeat = 0;
	//	for (int f = 10; f < 51; f += 10)
	//	{
	//		params.maxFeatures = f;
	//		LaserSpeckleVelocimeter LSV(path, Method::OPTICAL_FLOW, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		int newFeat = round(LSV.getFeatures() / static_cast<double>(frameNumber));
	//		if (newFeat != oldFeat)
	//		{
	//			oldFeat = newFeat;
	//			file << newFeat << ";";
	//			file << 1000 * sumTime << ";";
	//			file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//			file << abs(r.z) << ";";
	//			file << endl;
	//		}
	//	}
	//}
	//file.close();

	///*
	//Feature matching performance TRANSLATION
	//*/
	//path = mainPath + "\\gen10_2\\*.png";
	//file.open("featMatch10_extend.csv");
	//stringVec1 = { "BRISK", "ORB", "SURF", "U-SURF" };

	//for (stringIter1 = stringVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1)
	//{
	//	params.detector = *stringIter1;
	//	file << "Feat;" << *stringIter1 << "_time;" << *stringIter1 << "_acc;" << endl;

	//	int oldFeat = 0;
	//	for (int f = 10; f < 51; f += 10)
	//	{
	//		params.maxFeatures = f;
	//		LaserSpeckleVelocimeter LSV(path, Method::FEATURE_MATCHING, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		int newFeat = round(LSV.getFeatures() / static_cast<double>(frameNumber));
	//		if (newFeat != oldFeat)
	//		{
	//			oldFeat = newFeat;
	//			file << newFeat << ";";
	//			file << 1000 * sumTime << ";";
	//			file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//			file << endl;
	//		}
	//	}
	//}
	//file.close();

	///*
	//Feature matching performance ROTATION
	//*/
	//path = mainPath + "\\rot5_0\\*.png";
	//file.open("featMatchRot10_extend.csv");
	//stringVec1 = { "BRISK", "ORB", "SURF" };

	//for (stringIter1 = stringVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1)
	//{
	//	params.detector = *stringIter1;
	//	file << "Feat;" << *stringIter1 << "_time;" << *stringIter1 << "_acc;" << *stringIter1 << "_rotAcc;" << endl;

	//	int oldFeat = 0;
	//	for (int f = 10; f < 51; f += 10)
	//	{
	//		params.maxFeatures = f;
	//		LaserSpeckleVelocimeter LSV(path, Method::FEATURE_MATCHING, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		int newFeat = round(LSV.getFeatures() / static_cast<double>(frameNumber));
	//		if (newFeat != oldFeat)
	//		{
	//			oldFeat = newFeat;
	//			file << newFeat << ";";
	//			file << 1000 * sumTime << ";";
	//			file << sqrt(r.x*r.x + r.y*r.y) << ";";
	//			file << abs(r.z) << ";";
	//			file << endl;
	//		}
	//	}
	//}
	//file.close();

	///*
	//Different angles
	//*/
	//file.open("angles10.csv");
	//file << "Var;sFAST;sFAST_acc;sAGAST;sAGAST_acc;GFTT;GFTT_acc;Harris;Harris_acc;BRISK;BRISK_acc;ORB;ORB_acc\n";
	//stringVec1 = { "\\rot0_5\\*.png", "\\rot1_0\\*.png", "\\rot1_5\\*.png", "\\rot2_0\\*.png", "\\rot3_0\\*.png", "\\rot4_0\\*.png", "\\rot5_0\\*.png", "\\rot6_0\\*.png", 
	//	"\\rot7_0\\*.png", "\\rot8_5\\*.png", "\\rot10_0\\*.png", "\\rot20_0\\*.png", "\\rot30_0\\*.png" };
	//doubleVec1 = { 0.5, 1, 1.5,2, 3,4, 5,6, 7,8.5, 10, 20, 30 };
	//params.maxFeatures = 400;

	//stringVec2 = { "sFAST", "sAGAST", "GFTT", "Harris", "BRISK", "ORB" };

	//for (stringIter1 = stringVec1.begin(), doubleIter1 = doubleVec1.begin(); stringIter1 != stringVec1.end(); ++stringIter1, ++doubleIter1)
	//{
	//	file << *doubleIter1 << ";";
	//	path = mainPath + *stringIter1;

	//	for (stringIter2 = stringVec2.begin(); stringIter2 != stringVec2.end(); ++stringIter2)
	//	{
	//		params.detector = *stringIter2;
	//		if (*stringIter2 == "BRISK" || *stringIter2 == "ORB")
	//			method = Method::FEATURE_MATCHING;
	//		else
	//			method = Method::OPTICAL_FLOW;

	//		LaserSpeckleVelocimeter LSV(path, method, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 1)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);
	//		}

	//		sumTime /= (frameNumber - 1.0);

	//		auto r = evaluator.getAvgError();
	//		file << 1000 * sumTime << ";";
	//		file << abs(r.z) << ";";
	//	}

	//	file << endl;
	//}
	//file.close();

	/////*
	////Different images sizes for features matching
	////*/
	////ofstream featuresize("featuresize.csv");
	////featuresize << "Size;Flow_Speed;Flow_Acc;Brute_Matching_Speed;Brute_Matching_Acc\n";
	////for (auto i = 0; i < 4; i++)
	////{
	////	switch (i)
	////	{
	////	case 0: {path = mainPath + "\\gen161\\*.png"; featuresize << 161; break; }
	////	case 1: {path = mainPath + "\\gen240\\*.png"; featuresize << 240; break; }
	////	case 2: {path = mainPath + "\\gen320\\*.png"; featuresize << 320; break; }
	////	case 3: {path = mainPath + "\\gen400\\*.png"; featuresize << 400; break; }
	////	}
	////	featuresize << ";";

	////	for (int j = Method::OPTICAL_FLOW; j <= Method::FEATURE_MATCHING; j++)
	////	{
	////		params.detector = "FAST";
	////		if (j == Method::FEATURE_MATCHING) params.detector = "ORB";

	////		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	////		Evaluator evaluator(path);

	////		double sumTime = 0;
	////		uint64 frameNumber = 0;

	////		for (;;)
	////		{
	////			Mat frame;
	////			if (!LSV.nextMeasurement(frame)) break;

	////			auto V = LSV.getVelocity();
	////			auto displacement = LSV.getDisplacement();
	////			frameNumber = LSV.getFrameNumber();
	////			auto time = LSV.getTime();

	////			//accumulate results
	////			if (frameNumber > 2)
	////			{
	////				sumTime += time;
	////			}

	////			evaluator.evaluate(displacement, frameNumber);

	////			if (waitKey(1) != -1) break;
	////		}

	////		sumTime /= (frameNumber - 2.0);

	////		auto r = evaluator.getAvgError();
	////		featuresize << 1000 * sumTime << ";";
	////		featuresize << sqrt(r.x*r.x + r.y*r.y) << ";";
	////	}
	////	featuresize << endl;
	////}
	////featuresize.close();

	/*
	Different images sizes for features matching
	*/
	//params.metric = Metric::NXC;
	//ofstream imageSize("imageSize6.csv");
	//imageSize << "Size;FFT;FFT_Acc;LRP;LRP_Acc;Spiral;Spiral_Acc;Flow_Speed;Flow_Acc;Brute_Matching_Speed;Brute_Matching_Acc\n";
	//for (auto i = 0; i < 5; i++)
	//{
	//	switch (i)
	//	{
	//	case 0: {path = mainPath + "\\gen10_2_160\\*.png"; imageSize << 160; break; }
	//	case 1: {path = mainPath + "\\gen10_2_256\\*.png"; imageSize << 256; break; }
	//	case 2: {path = mainPath + "\\gen10_2_400\\*.png"; imageSize << 400; break; }
	//	case 3: {path = mainPath + "\\gen10_2\\*.png"; imageSize << 512; break; }
	//	case 4: {path = mainPath + "\\gen10_2_768\\*.png"; imageSize << 768; break; }
	//	}
	//	imageSize << ";";

	//	for (int j = 0; j <= Method::SPIRAL; j++)
	//	{
	//		params.detector = "FAST";
	//		if (j == Method::FULL_SPATIAL) continue;
	//		if (j == Method::OPTICAL_FLOW)
	//		{
	//			params.maxFeatures = 480;
	//			params.detector = "sFAST";
	//		}
	//		if (j == Method::FEATURE_MATCHING)
	//		{
	//			params.maxFeatures = 500;
	//			params.detector = "BRISK";
	//		}

	//		LaserSpeckleVelocimeter LSV(path, j, params, px2mm, draw);

	//		Evaluator evaluator(path);

	//		double sumTime = 0;
	//		uint64 frameNumber = 0;

	//		for (;;)
	//		{
	//			Mat frame;
	//			if (!LSV.nextMeasurement(frame)) break;

	//			auto V = LSV.getVelocity();
	//			auto displacement = LSV.getDisplacement();
	//			frameNumber = LSV.getFrameNumber();
	//			auto time = LSV.getTime();

	//			//accumulate results
	//			if (frameNumber > 2)
	//			{
	//				sumTime += time;
	//			}

	//			evaluator.evaluate(displacement, frameNumber);

	//		}

	//		sumTime /= (frameNumber - 2.0);

	//		auto r = evaluator.getAvgError();
	//		imageSize << 1000 * sumTime << ";";
	//		imageSize << sqrt(r.x*r.x + r.y*r.y) << ";";
	//	}
	//	imageSize << endl;
	//}
	//imageSize.close();

	return EXIT_SUCCESS;
}
