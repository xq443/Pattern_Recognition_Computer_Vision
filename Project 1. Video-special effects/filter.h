#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp> 

void applyGrayscale(cv::Mat& img);
int greyscale(cv::Mat &src, cv::Mat &dst);

#endif
