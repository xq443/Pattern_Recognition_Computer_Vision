/*
  Xujia Qin
  January 15th 2025
  CS 5330 OpenCV Project 1 
 */

#ifndef TIMEBLUR_H
#define TIMEBLUR_H

#include <opencv2/opencv.hpp>

// Declare the blur function
int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2(cv::Mat& src, cv::Mat& dst);

#endif // TIMEBLUR_H
