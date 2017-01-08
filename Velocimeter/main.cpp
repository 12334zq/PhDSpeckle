/**
LSVelocimeter project
@author Krzysztof Kotowski
*/
#include "LaserSpeckleVelocimeter.h"
#include "Evaluator.h"

using namespace std;

int main(int argc, char** argv)
{
	cout << "Speckle velocimetry - Krzysztof Kotowski\n";

	int method = Method::OPTICAL_FLOW;
	bool draw = true;
	bool evaluate = true;
	double px2mm = 1.0;

	MethodParams params;
	params.metric = Metric::NXC;
	params.templRatio = 0.8;
	params.maxShift = 0.1;
	params.layers = 3;
	params.detector = MyFeature2D::FAST;
	params.maxFeatures = 200;
	params.RANSAC = true;
	params.matcher = "BruteForce";

	String path = "C:\\Users\\Krzysztof\\Pictures\\Speckle\\gen10_2\\*.png";
	//String path = "C:\\Users\\Krzysztof\\Pictures\\Speckle\\rot1_0\\*.png";
	//String path = "C:\\Users\\Krzysztof\\Pictures\\Speckle\\rotations\\*.png";
	//String path = "C:\\Users\\Krzysztof\\Pictures\\Speckle\\rot30_0\\*.png";
	//String path = "0";
	//String path = "C:\\Users\\Krzysztof\\Pictures\\Speckle\\out.mp4";

	try
	{
		stoi(path);
		evaluate = false;
	}catch(...)
	{
		
	}

	/**
	* CHECK ARGUMENTS
	*
	*/
	if (argc < 12)
	{
		cout << "Usage : %s <source> <method> (<metric> <templRatio> <maxShift> <layers> <detector> <RANSAC> <matcher> <draw> <px2mm>)\n", argv[0];
	}

	if (argc > 1) path = argv[1];
	if (argc > 2) method = stoi(argv[2]);
	if (argc > 3) params.metric = stoi(argv[3]);
	if (argc > 4) params.templRatio = stod(argv[4]);
	if (argc > 5) params.maxShift = stod(argv[5]);
	if (argc > 6) params.layers = stoi(argv[6]);
	if (argc > 7) params.detector = stoi(argv[7]);
	if (argc > 8) params.maxFeatures = stoi(argv[8]);
	if (argc > 9) params.RANSAC = stoi(argv[9]);
	if (argc > 10) params.matcher = argv[10];
	if (argc > 11) draw = stoi(argv[11]);
	if (argc > 12) px2mm = stod(argv[12]);
	String px_mm = "px";
	if (px2mm > 1.0 + DBL_EPSILON && px2mm < 1.0 - DBL_EPSILON)
		px_mm = "mm";

	LaserSpeckleVelocimeter LSV(path, method, params, px2mm, draw);

	if (path == "0")
		evaluate = false;

	Evaluator evaluator;
	if (evaluate) evaluator = Evaluator(path);
	evaluate = evaluator.getStatus();

	double sumTime = 0;
	Point2f sumVelocity(0);
	uint64 frameNumber = 0;

	cout << "Ready. Push Enter to start\n";
	cin.ignore();
	waitKey();

	double timeAccDelay = 5.0;
	namedWindow("Frame", WINDOW_NORMAL);
	resizeWindow("Frame", 700, 700);

	for (;;)
	{
		Mat frame;
		if (!LSV.nextMeasurement(frame)) break;

		auto V = LSV.getVelocity();
		auto displacement = LSV.getDisplacement();
		frameNumber = LSV.getFrameNumber();
		auto time = LSV.getTime();

		//accumulate results
		if (frameNumber > timeAccDelay)
		{
			sumTime += time;
		}
		if (frameNumber > 1)
			sumVelocity += V;

		if (evaluate) evaluator.evaluate(displacement, frameNumber);

		//display result
		cout << frameNumber << ". calculated in " << 1000 * time << "ms\n"
			<< "Translation: ( " << displacement.x << ", " << displacement.y << " ) " << px_mm << " | Velocity: " << V.x << " " << px_mm << "/s\n";
		if (V.y > 0) cout << "Rotation: " << displacement.z << " deg  |  Velocity: " << V.y << " deg/s\n";
		cout << endl;

		if (draw)
		{
			imshow("Frame", LSV.getResultImg());
		}
		else
			imshow("Frame", frame);

		if (waitKey(1) != -1) break;
	}

	sumVelocity /= (frameNumber - timeAccDelay);
	sumTime /= (frameNumber - timeAccDelay);
	cout << endl << "Average velocity: " << sumVelocity.x << " " << px_mm << "/s";
	if (sumVelocity.y > 0) cout << "  |  " << sumVelocity.y << " deg/s";
	cout << endl;

	if (evaluator.getStatus())
	{
		//get directory name from path
		size_t pos = path.find_last_of('\\');
		if (pos == string::npos) pos = path.find_last_of('/');
		if (pos == string::npos) CV_Error(CV_StsBadArg, "Directory not found\n");
		string groundPath(path);
		groundPath.resize(pos + 1);

		ofstream result(groundPath + "res" + to_string(method) + to_string(params.metric) + to_string((int)(params.templRatio * 100)) + to_string(int(params.maxShift * 100)) + to_string(params.layers) + to_string(params.detector) + to_string(params.RANSAC) + params.matcher + ".txt");

		auto r = evaluator.getAvgError();
		cout << "Avg. error: " << evaluator.getAvgError() << endl;
		cout << "Avg. error in pixels: " << sqrt(r.x*r.x + r.y*r.y) << endl;
		result << "Avg. error: " << evaluator.getAvgError() << endl;
		result << "Avg. error in pixels: " << sqrt(r.x*r.x + r.y*r.y) << endl;
		result << "Avg. time: " << 1000 * sumTime << " ms\n";
		//namedWindow("Error", WINDOW_NORMAL);
		//imshow("Error", evaluator.getPathImg());

		result.close();
	}
	cout << "Avg. time: " << 1000 * sumTime << " ms\n";

	waitKey();

	return EXIT_SUCCESS;
}
