/*
	James Marcel

	Runs webcam and processes each frame into cleaned up binary image
	with regions identified, then chooses a region and compares it to other objects 
	in the database, then overlays the nearest match in window
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "recog.h"
#include "csv_util.h"



//Run webcam and processes binary image for each frame,
//cleans it, runs segmentation (region growing) and then
//calculates scale/translation/rotation invariant feature vectors
//pressing 'n' key will capture the feature vector for the current frame with user input as object name
int main(int argc, char* argv[]) {

	bool knn = false;
	int k = 3; //default k value
	std::vector<char*> objNames;
	std::vector<std::vector<float>>objData;
	char csvFile[] = "object_database";

	float devs[2] = { 0 };  //stores std dev for the two invariant features used in distance calculation
	read_image_data_csv(csvFile, objNames, objData, 0);
	
	deviation(objData, devs);  //calculate std dev for database features

	//getting K from arguments if provided
	if (argc > 1) {
		if (std::stoi(argv[1]) < 1 || std::stoi(argv[1]) > 5) {
			printf("For K-nearest neighbors, please provide an integer value from 1 to 5.\n");
		}
		else {
			knn = true;
			k = std::stoi(argv[1]);
			printf("Using %d-Nearest Neighbors\n", k);
		}
	}
	else {
		printf("Using nearest neighbor.\n");
	}



	//opening video device
	cv::VideoCapture* capdev;
	capdev = new cv::VideoCapture(0);
	if (!capdev->isOpened()) {
		printf("Unable to open video device\n");
		return -1;

	}

	//get some properties of the image
	cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
		(int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
	printf("Expected size: %d %d \n", refS.width, refS.height);

	cv::namedWindow("Video", 1); //identifies a window
	cv::Mat frame;




	for (;;) {
		*capdev >> frame;
		if (frame.empty()) {
			printf("frame is empty\n");
			break;
		}

		//see if there is a keystroke
		char key = cv::waitKey(10);
		if (key == 'q') {
			break;
		}

		cv::imshow("Video",frame);

		cv::Mat bImg;  //used to store binary image

		binaryImg(frame, bImg, 120);
		cv::imshow("Binary/Threshold", bImg);

		cv::Mat distance; //used to store distance matrix
		grassfire(bImg, distance); //calculating distance on binary img

		cv::Mat eroded; //stores eroded/dilated image
		distErosion(distance, eroded, 2);

		cv::Mat final;
		cv::Mat temp;

		int x = 6;
		//repeats dilation x number of times
		for (int i = 0; i < x; i++) {
			dilate(eroded, final);
			eroded = final.clone();
		}

		cv::imshow("Clean-up", eroded);

		cv::Mat regtest;

		//finding regions
		int regnum = regions(final, regtest);

		//finding majority region in center of image
		int central = centralRegion(regtest);

		//calculate raw moments for this region (M10 avg x, M01 avg y, M00 total pix)
		int moments[3] = { 0 };
		rawMoments(regtest, central, moments);



		cv::Mat tester;
		//gives each region a different color
		regColor(regtest, tester, regnum);
		//adds littl red cross to center of object
		objCenter(tester, moments);


		//mu stores the feature vector for each frame: mu 20, mu 02, mu 11, angle alpha, angle beta, mu 22, fill %, h/w ratio
		double mu[8] = { 0 }; 
		//calculating invariant moments
		angleAlpha(regtest, central, moments, mu);
		invarMoment(regtest, central, moments, mu);

		//used for calculating degrees from radians
		const double deg = 180 / 3.14159265358979323846;
		double tilt = mu[4] * deg;

		//rotating image - first get rotation matrix
		cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(static_cast<float>(moments[0]), static_cast<float>(moments[1])), tilt, 1);

		cv::Mat rotatedFinal;
		cv::Mat rotatedRegion;

		//then warp based on that matrix
		//warp uses no flags so region values aren't affected by the algo
		warpAffine(final, rotatedFinal, rotation, final.size(), 0, cv::BORDER_TRANSPARENT);
		warpAffine(regtest, rotatedRegion, rotation, final.size(), 0, cv::BORDER_TRANSPARENT);

		//getting bounding box for region
		int box[4] = { 0 };
		getBox(rotatedRegion, central, box);
		getRatio(rotatedRegion, central, box, mu);

		//assigning points from calculated bounding box
		cv::Point topleft(box[0], box[2]);
		cv::Point botright(box[1], box[3]);

		//drawing Oriented Bounding Box
		cv::cvtColor(rotatedFinal, final, cv::COLOR_GRAY2BGR);
		objCenter(final, moments);
		cv::rectangle(final, topleft, botright, cv::Scalar(0, 0, 255), 1);

		//showing regions and post cleanup binary image
		cv::imshow("Regions", tester);


	
		//processing distance to already classified objects
		char result[256];

		if (knn == true) { //if k parameter provided, use k-nearest neighbors
			kNearest(mu, devs, objData, objNames, result, k);
		}
		else {  //otherwise use nearest neighbor
			nearestNeighb(mu, devs, objData, objNames, result);
		}
		
		

		//making strings for live feature display
		std::string feature = "fill %: " +  std::to_string(mu[6]);
		std::string feature2 = "h/w ratio: " + std::to_string(mu[7]);

		//adding text overlays to final frame
		cv::putText(final,result,cv::Point(40,final.rows - 40),1,5,cv::Scalar(255,0,0));
		cv::putText(final, feature, cv::Point(final.cols - 350, 30), 2, 1, cv::Scalar(0, 0, 255));
		cv::putText(final, feature2, cv::Point(final.cols - 350, 70), 2, 1, cv::Scalar(0, 0, 255));

		cv::imshow("OBB/Center", final);	 //displaying final processed frame

	
		//Feature are written to database by pressing n key
		//name of the object must then be entered into console
		if (key == 'n') {
			char input[256];

			std::cin >> input;

			//writing regTest to store feature vector with object name
			char fname[] = "object_database";

			append_image_data_csv(fname, input, mu, false);

		}
		

	}

	
	return 0;

}