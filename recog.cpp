/*
	James Marcel

	Binary image cleanup and feature vector functions
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <dirent.h>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include "recog.h"


//calculates which object is closest to the target based on 
//the sum of distances from the k-nearest neigbors of each object class to the target
int kNearest(double* target, float* dev, std::vector<std::vector<float>> data, std::vector<char*> objNames, char* result, int k) {
	if (k > 4) { //I only captured 4 different sets of features for each object
		k = 4;
	}
	else if (k < 1) {
		k = 1;
	}

	std::map<std::string, std::vector<std::vector<float>>> objList; //map to store vectors based on object

	float distFill = 0;
	float distShape = 0;
	float sum = 0;

	//adding each feature vector to a map with its name as a key
	for (int i = 0; i < data.size(); i++) {
		std::string name(objNames[i]);

		if (!objList.count(objNames[i])) {

			std::vector<std::vector<float>> temp;
			objList[name] = temp;

		}
		objList[name].push_back(data[i]);

	}


	float topScore = 99999;
	std::string topObj;
	std::vector<float> objDist;

	for (const auto& x : objList) {  //for each element x of objList map

		objDist.clear();

		//calculating all distances for each object
		for (int j = 0; j < x.second.size(); j++) {

			float fill = x.second[j][6];
			float shape = x.second[j][7];

			distFill = ((target[6] - fill) / dev[0]) * ((target[6] - fill) / dev[0]);
			distShape = ((target[7] - shape) / dev[1]) * ((target[7] - shape) / dev[1]);
			sum = distFill + distShape; //distance from this object to target

			objDist.push_back(sum); //adding this distance to object's distances

		}
		
		//finding the k closest by sorting and summing top k distances
		//distSort(objDist);

		sort(objDist.begin(), objDist.end());

		float score = 0;
		for (int z = 0; z < k; z++) {
			score += objDist[z];
		}

		//update best option if this is best option
		if (score < topScore) {
			topScore = score;
			topObj = x.first;

		}

	}

	strcpy(result, topObj.c_str());


	return 0;
}




//calculates distance between database features and target.
//Returns name of lowest distance object
int nearestNeighb(double* target, float* dev, std::vector<std::vector<float>> data, std::vector<char*> objNames, char* result) {

	int lowestPlace = 0;
	float lowestVal = 99999;

	float distFill = 0;
	float distShape = 0;
	float sum = 0;

	for (int i = 0; i < data.size(); i++) {

		//calculate distance from target to db feature
		distFill = ((target[6] - data[i][6]) / dev[0]) * ((target[6] - data[i][6]) / dev[0]);
		distShape = ((target[7] - data[i][7]) / dev[1]) * ((target[7] - data[i][7]) / dev[1]);

		sum = distFill + distShape;

		if (sum < lowestVal) { //save nearest neighbor position/value
			lowestVal = sum;
			lowestPlace = i;
		}

	}

	strcpy(result, objNames[lowestPlace]);

	return 0;
}


//calculates stddev for invariant features (fill ratio and h/w ratio)
int deviation(std::vector<std::vector<float>> data, float* result) {

	float sumFill = 0;
	float sumShape = 0;
	float fillAvg = 0;
	float shapeAvg = 0;

	//iterate through each object in db for averages
	for (int i = 0; i < data.size(); i++) {

		sumFill += data[i][6]; //position of each OBB fill percentage
		sumShape += data[i][7]; //pos of each height/width ratio

	}

	fillAvg = sumFill / data.size();
	shapeAvg = sumShape / data.size();

	float sseFill = 0;
	float sseShape = 0;

	for (int i = 0; i < data.size(); i++) {
		//sum squared difference
		sseFill += (data[i][6] - fillAvg) * (data[i][6] - fillAvg); 
		sseShape += (data[i][7] - shapeAvg) * (data[i][7] - shapeAvg);

	}


	result[0] = sqrt(sseFill / data.size());
	result[1] = sqrt(sseShape / data.size());


	return 0;
}







//Adds both h/w ratio and fill percentage to feature vector
int getRatio(cv::Mat& src,int region, int* obb, double* feature) {

	//iterate only through bounding box
	int rstart = obb[2];
	int rend = obb[3];
	int cstart = obb[0];
	int cend = obb[1];

	double count = 0;
	double inRegion = 0;
	int test = 0;

	//counting empty and full pixels
	for (int i = rstart; i <= rend; i++) {

		uchar* rptr = src.ptr<uchar>(i);

		for (int j = cstart; j <= cend; j++) {
			
			count += 1;

			if (rptr[j] == region) {
				inRegion += 1;
			}
		}
	}


	//calculate fill percentage
	double fill = inRegion / count;
	feature[6] = fill;

	double height = obb[3] - obb[2];
	double width = obb[1] - obb[0];

	//this h/w ratio calculation always divides by longer dimension
	if (width > height) {
		feature[7] = height / width;
	}
	else {
		feature[7] = width / height;
	}
	

	//printf("the fill percentage is %f and the h/w ratio is %f\n", feature[6], feature[7]);


	return 0;
}




//returns 4 coordinates that bound object
//takes a mat with region values and an array to fill up
int getBox(cv::Mat& src, int region, int* box ) {

	box[0] = src.cols;  //x min
	box[1] = 0; //x max
	box[2] = src.rows;  //y min
	box[3] = 0; //y max


	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);

		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] == region){

				if (j < box[0]) { //if x is more left than current x min
					box[0] = j;
				}
				if (j > box[1]) { //if x is more right than current x max
					box[1] = j;
				}
				if (i < box[2]) { //if y is higher than current y min
					box[2] = i;
				}
				if (i > box[3]) { //if y is lower than current y max
					box[3] = i;
				}
			}
		}
	}

	//printf("Coordinates are %d,%d, %d and %d\n", box[0], box[1], box[2], box[3]);

	return 0;
}




//Extension 3: Invariant Moments Calculation
//calculates mu22 of the given region based on previously calculated moments
int invarMoment(cv::Mat& src, int region, int* moments, double* mumoments) {

	double cosB = cos(mumoments[4]);//constants for formula
	double sinB = sin(mumoments[4]);

	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);

		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] == region) {
				//mu22 formula
				mumoments[5] += (((i - moments[1]) * cosB) + ((j - moments[0]) * sinB))
					* (((i - moments[1]) * cosB) + ((j - moments[0]) * sinB)); 

			}
		}
	}

	//normalizing to number of pixels
	double totalPix = 1 / static_cast<double>(moments[2]);
	double temp = mumoments[5];
	mumoments[5] = temp * totalPix;


	return 0;

}


//calculates mu20 mu02 and mu11 in order to get angle alpha
int angleAlpha(cv::Mat& src,int region, int* moments, double* mumoments) {
	

	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);

		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] == region) {

				mumoments[0] += (j - moments[0]) * (j - moments[0]); //mu20 (x)
				mumoments[1] += (i - moments[1]) * (i - moments[1]); //mu02 (y)
				mumoments[2] += (j - moments[0]) * (i - moments[1]); //mu11 (x and y)

			}
		}
	}

	//printf("pre-normalized m20 is %f, m02 is %f, m11 is %f\n", mumoments[0], mumoments[1], mumoments[2]);

	double temp = 0;
	double normal = 1 / static_cast<double>(moments[2]);
	for (int x = 0; x < 3; x++) {
		temp = mumoments[x];
		mumoments[x] = temp * normal;
	}

	const double pi2 = 1.57079632679489661923;
	double angle = 0.5 * atan(2 * mumoments[2] / (mumoments[0] - mumoments[1]));
	mumoments[3] = angle;
	mumoments[4] = angle + pi2;

	//printf("m20 is %f, m02 is %f, m11 is %f, alpha angle is %f, beta is %f\n", mumoments[0], mumoments[1], mumoments[2], mumoments[3], mumoments[4]);

	return 0;

}



//places red cross at center of central region ( takes color src )
int objCenter(cv::Mat &src, int* moments) {

	cv::Vec3b* row = src.ptr<cv::Vec3b>(moments[1]);//ptr to row (y) of center
	cv::Vec3b* top = src.ptr<cv::Vec3b>(moments[1] - 1); //above and below
	cv::Vec3b* top2 = src.ptr<cv::Vec3b>(moments[1] - 2);
	cv::Vec3b* bottom = src.ptr<cv::Vec3b>(moments[1] + 1);
	cv::Vec3b* bottom2 = src.ptr<cv::Vec3b>(moments[1] + 2);

	int x = 0;
	for (int c = 0; c < 3; c++) {
		if (c == 2) { //only set red value to max
			x = 255;
		}
		else {
			x = 100;
		}
		row[moments[0] - 2][c] = x;
		row[moments[0] - 1][c] = x;
		row[moments[0]][c] = x;
		row[moments[0] + 1][c] = x;
		row[moments[0] + 2][c] = x;
		top[moments[0]][c] = x;
		top2[moments[0]][c] = x;
		bottom[moments[0]][c] = x;
		bottom2[moments[0]][c] = x;
	}

	

	return 0;
}


//calculate raw moments for this region (M10 avg x, M01 avg y, M00 total pix)
//accepts pointer to an 3 element int array
int* rawMoments(cv::Mat& src, int region, int* moments) {

	//static int moments[3] = { 0 };

	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);

		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] == region){

				moments[0] += j; //summing all x values
				moments[1] += i; //summing all y values
				moments[2] += 1; //count of all pixels in region
				
			}
		}
	}

	int tempx = moments[0];
	int tempy = moments[1];
	moments[0] = tempx / moments[2]; //sets average x value
	moments[1] = tempy / moments[2]; //sets average y value

	//printf("x center is %d, y center is %d and total pix is %d\n",moments[0],moments[1],moments[2]);
	return moments;
}



//only iterate through central third of given Mat
//to find majority region within center of image
//returns the integer value given to that region in the region map
int centralRegion(cv::Mat &src) {
	int region = 0;

	std::map<int, int> regCount; //stores pix quantity of each region found

	//only iterate through central third of given Mat
	//to find majority region in this zone
	int rstart = src.rows / 3;
	int rend = src.rows - (src.rows / 3);
	int cstart = src.cols / 3;
	int cend = src.cols - (src.cols / 3);

	for (int i = rstart; i < rend; i++) {

		uchar* rptr = src.ptr<uchar>(i);

		for (int j = cstart; j < cend; j++) {

			if (rptr[j] != 0) {

				regCount[rptr[j]] += 1; //add this value to map structure and increment
			}

		}
	}

	//iterate through all region counts to find max
	int maxReg = 0;
	for (const auto& x : regCount) {
		//printf("Region %d has %d pixels\n", x.first, x.second);
		if (x.second > maxReg){
			maxReg = x.second;
			region = x.first;
		}
	}


	return region;
}



//creates a color coded image from connected region map 
int regColor(cv::Mat& src, cv::Mat& dst, int regCount) {

	dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);

	//map to store color values
	std::map<int, std::tuple<int, int, int>> color;
	//assigning as many random values as regions
	for (int x = 0; x < regCount; x++) {

		int b = (255 / regCount) * x;
		int g = 255 - (255 / regCount) * x;
		int r = rand() % 255;
		color[x] = std::make_tuple(b, g, r);


		//printf("color for region %d is %d\n", x, color[x]);
	}
	

	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);
		cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);

		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] == 0) {
				dptr[j][0] = 255;
				dptr[j][1] = 255;
				dptr[j][2] = 255;
			}
			else { //otherwise set destination color to map value for this region
				int value = static_cast<int>(rptr[j]);
				dptr[j][0] = std::get<0>(color[value]);
				dptr[j][1] = std::get<1>(color[value]);
				dptr[j][2] = std::get<2>(color[value]);
			}
		}
	}

	return 0;
}




//Extension 2: Region Growing Function
//fills destination mat with 0s for background and #s to indicate 
//connected regions for foreground pixels
//returns number of regions counted
int regions(cv::Mat& src, cv::Mat&dst) {

	dst = cv::Mat::zeros(src.size(), CV_8UC1);

	//stack is vector of tuples. Each one holds row/col coordinates
	std::vector<std::tuple<int, int>> stack; 
	std::tuple<int, int> pixel;

	int region = 1; //init region IDs

	//loop looking for 'seed' pixels
	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);
		uchar* dptr = dst.ptr<uchar>(i);

		for (int j = 0; j < src.cols; j++) {

			//if this pixel is in fg and region unassigned
			if ((rptr[j] == 0) && (dptr[j] == 0)) {
				dptr[j] = region;
				
				//make tuple from coords
				std::tuple<int, int> temp;
				temp = std::make_tuple(i, j);
				stack.push_back(temp); //add it to stack

				while (stack.size() > 0) { //scan region until complete
					//pop last-in pixel
					pixel = stack.back();
					stack.pop_back();
					//getting coordinates from tuple
					int x = std::get<0>(pixel);
					int y = std::get<1>(pixel);

					//pointers for checking src and dest neighbors
					uchar* rptr2 = src.ptr<uchar>(x);
					uchar* dptr2 = dst.ptr<uchar>(x);
					uchar* rtptr2 = src.ptr<uchar>(x-1);
					uchar* dtptr2 = dst.ptr<uchar>(x-1);
					uchar* rbptr2 = src.ptr<uchar>(x+1);
					uchar* dbptr2 = dst.ptr<uchar>(x+1);

					//check boundary before each neighbor check
					if (y != 0) {//left
						
						//each neighbor needs to be in foreground and region unlabelled

						if ((rptr2[y - 1] == 0) && (dptr2[y - 1] == 0)) {
							//printf("then got to comparison\n");
							dptr2[y - 1] = region;
							stack.push_back({ x, y - 1});
						}
					}
					if (y != src.cols - 1) { //right

						if ((rptr2[y + 1] == 0) && (dptr2[y + 1] == 0)) {
							dptr2[y + 1] = region;
							stack.push_back({ x, y + 1 });
						}
					}
					if (x != 0) { //top

						if ((rtptr2[y] == 0) && (dtptr2[y] == 0)) {
							dtptr2[y] = region;
							stack.push_back({ x - 1, y});
						}
					}
					if (x != src.rows - 1) { //bottom

						if ((rbptr2[y] == 0) && (dbptr2[y] == 0)) {
							dbptr2[y] = region;
							stack.push_back({ x + 1, y  });
						}
					}
				}
				region++;
			}
		}
	}

	//printf("ended with %d regions\n", region);

	return region;
}



//Extension 1: Dilation function from scratch
//grows pixels with 8-connected pattern
int dilate(cv::Mat& src, cv::Mat& dst) {

	//fills destination with white pixels
	dst = cv::Mat::ones(src.size(), CV_8UC1)*255;


	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i); //source pointer
		uchar* dptr = dst.ptr<uchar>(i); //destination pointer
		uchar* tptr = src.ptr<uchar>(i);
		uchar* bptr = src.ptr<uchar>(i);
		uchar* dtptr = dst.ptr<uchar>(i);
		uchar* dbptr = dst.ptr<uchar>(i);
		if (i > 0) { //only get these pointers if they're valid locations
			tptr = src.ptr<uchar>(i-1);//src row above
			uchar* dtptr = dst.ptr<uchar>(i-1); //destination row above
		}
		if (i < src.rows - 1) {
			bptr = src.ptr<uchar>(i+1);//src row below
			uchar* dbptr = dst.ptr<uchar>(i+1); //destination row below
		}
		

		for (int j = 0; j < src.cols; j++) {

			//don't need to operate on src background pixels
			if (rptr[j] == 255) {  //if already filled, ignore

			}
			//dilating every black pixel from here on
			else if (j == 0) { //left edge

				dtptr[j] = 0;
				dtptr[j + 1] = 0;
				dptr[j + 1] = 0;
				dbptr[j] = 0;
				dbptr[j + 1] = 0;
			}
			else if (j == src.cols - 1) { //right edge
				dtptr[j - 1] = 0;
				dtptr[j] = 0;
				dptr[j - 1] = 0;
				dbptr[j - 1] = 0;
				dbptr[j] = 0;

			}
			else { //assign 8 surrounding pixels


				dtptr[j - 1] = 0;
				dtptr[j] = 0;
				dtptr[j + 1] = 0;
				dptr[j - 1] = 0;
				dptr[j + 1] = 0;
				dbptr[j - 1] = 0;
				dbptr[j] = 0;
				dbptr[j + 1] = 0;

			}
		}
	}

	return 0;
}



//Extension 1: Grassfire Algorithm
//grassfire algorithm to calculate distances from background for each pixel
//fills out a matrix of int values with manhattan dist of each pix to background
int grassfire(cv::Mat& src, cv::Mat& distance) {

	//everything starts with distance 0
	distance = cv::Mat::zeros(src.size(), CV_8UC1);

	std::vector<std::vector<int>> dist(src.rows, std::vector<int>(src.cols,0)); //zeroing a vector of ints the same size as src image

	int countbg = 0;
	int countfg = 0;
	//first pass of grassfire algo
	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = src.ptr<uchar>(i);  //row pointer for binary src image
		//uchar* dptr = distance.ptr<uchar>(i);  //row pointer for distance matrix
		//uchar* dptr2 = distance.ptr<uchar>(i);  //row pointer for distance matrix


		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] == 255) {  //keep bg values as 0
				//dptr[j] = 0;
				dist[i][j] = 0;
				countbg++;
			}
			else { //otherwise use minimum of up or left pixels + 1
				if (i == 0) { //first row gets all 1s for foreground
					//dptr[j] = 1;
					dist[i][j] = 1;
				}
				else if (j == 0) { //first col gets all 1s for foreground
					//dptr[j] = 1;
					dist[i][j] = 1;
				}
				
				else {
					countfg++;
					if (dist[i][j - 1] > dist[i-1][j]) {
						dist[i][j] = dist[i-1][j] + 1;
						
					}
					else {
						dist[i][j] = dist[i][j-1] + 1;
						
					}

				}
			}
		}
	}
	
	//2nd pass starts from bottom right to top left
	for (int i = distance.rows - 1; i >= 0; i--) {


		uchar* dptr = distance.ptr<uchar>(i);  //row pointer for distance matrix

		for (int j = distance.cols - 1; j >= 0; j--) {
			
			if (dist[i][j] != 0) {  //for all foreground pixels
				//right and bottom edge cases
				if ((j == distance.cols - 1) || (i == distance.rows - 1)) {
					dist[i][j] = 1;
				}
				else {

					//if up neighbors + 1 are min and less than current value
					if ((dist[i][j + 1] > dist[i + 1][j]) && (dist[i + 1][j] + 1 < dist[i][j])) {
						dist[i][j] = dist[i + 1][j] + 1;
					}
					//if right neighbors +1 are min and less than current value
					else if (dist[i][j + 1] + 1 < dist[i][j]) {
						dist[i][j] = dist[i][j + 1] + 1;
					}
				}

			}

			//copying result to mat format
			if (dist[i][j] > 255) {
				dptr[j] = 255;
			}
			else {
				dptr[j] = dist[i][j];
			}
			
		}
	}


	//printf("%d in foreground, %d in background\n", countfg, countbg);

	return 0;
}


//Extension 1: erosion from Grassfire distance map
//erodes src image to destination based on distances in distance matrix up to level
int distErosion(cv::Mat &distance, cv::Mat & dst, int level) {

	dst = cv::Mat::zeros(distance.size(), CV_8UC1);

	for (int i = 0; i < distance.rows; i++) {

		uchar* rptr = distance.ptr<uchar>(i);  //row pointer for binary src image
		uchar* dptr = dst.ptr<uchar>(i);  //row pointer for binary src image

		for (int j = 0; j < distance.cols; j++) {

			if (j % 100 == 0 && i % 100 == 0) {
				//printf("row %d col %d = %d\n", i, j, rptr[j]);
			}
			if (rptr[j] < level) { // if distance to bg is less than level add to bg
				dptr[j] = 255;
			}
			else {  //otherwise set as foreground
				dptr[j] = 0;
			}
		}
	}

	return 0;
}



//generates a binary image with values of 0 or 255 based on grayscale values hitting threshold (thresh)
int binaryImg(cv::Mat &src, cv::Mat &dst, int thresh) {


	cv::Mat gray;
	cv::cvtColor(src, gray, CV_16F);  //making grayscale copy of src to work with

	dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1); //unsigned char datatype

	for (int i = 0; i < src.rows; i++) {

		uchar* rptr = gray.ptr<uchar>(i);  //row pointer for grayscale src image
		uchar* dptr = dst.ptr<uchar>(i);  //row pointer for grayscale src image

		for (int j = 0; j < src.cols; j++) {

			if (rptr[j] > thresh) {   //any values over thresh are set to 255 for binary image
				dptr[j] = 255;
			}
		}
	}


	return 0;
}



