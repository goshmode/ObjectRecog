/*
	James Marcel

	header for binary image feature vector functions
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>


//generates a binary image of 0 or 255 based on grayscale values hitting threshold (thresh)
int binaryImg(cv::Mat& src, cv::Mat& dst, int thresh);

//Extension 1
//fills out a matrix of int values with manhattan dist of each pix to background in 4-connected pattern
int grassfire(cv::Mat& src, cv::Mat& distance);

//Extension 1
//erodes src image to destination based on distances in distance matrix up to level
int distErosion(cv::Mat& distance, cv::Mat& dst, int level);

//Extension 1
//grows pixels with 8-connected pattern
int dilate(cv::Mat& src, cv::Mat& dst);

//Extension 2
//fills destination mat with 0s for background and #s to indicate 
//connected regions for foreground pixels
int regions(cv::Mat& src, cv::Mat& dst);

//creates a color coded image from connected region map 
int regColor(cv::Mat& src, cv::Mat& dst, int regCount);

//iterates through central third of given Mat
//to find majority region within center of image
//returns the integer value given to that region in the region map
int centralRegion(cv::Mat& src);

//Extension 3
//calculate raw moments for this region (M10 avg x, M01 avg y, M00 total pix)
int* rawMoments(cv::Mat& src, int region, int* moments);

//places red cross at center of central region ( takes color src )
int objCenter(cv::Mat& src, int* moments);

//calculates mu20 mu02 and mu11 in order to get angle alpha
int angleAlpha(cv::Mat& src, int region, int* moments, double* mumoments);

//extension 3
//calculates mu22 of the given region based on previously calculated moments
int invarMoment(cv::Mat& src, int region, int* moments, double* mumoments);

//returns 4 coordinates that bound object
//takes a mat with region values and an array to fill up
int getBox(cv::Mat& src, int region, int* box);

//Adds both h/w ratio and fill percentage to feature vector
int getRatio(cv::Mat& src, int region, int* obb, double* feature);


//calculates stddev for invariant features and calculates distance between
//database features and target.
//Returns name of lowest distance object
int nearestNeighb(double* target, float* dev, std::vector<std::vector<float>> data, std::vector<char*> objNames, char* result);

//calculates stddev for invariant features (fill ratio and h/w ratio)
int deviation(std::vector<std::vector<float>> data, float* result);


//calculates which object is closest to the target based on 
//the sum of distances from the k-nearest neigbors of each object class to the target
int kNearest(double* target, float* dev, std::vector<std::vector<float>> data, std::vector<char*> objNames, char* result, int k);
