/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1 task 4
 */

#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp> 

void applyGrayscale(cv::Mat& img);
int greyscale(cv::Mat &src, cv::Mat &dst);
void applySepiaTone(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
void cartoonEffect(cv::Mat &src, cv::Mat &dst);
void embossEffect(cv::Mat &src, cv::Mat &dst);
void applySepiaToneWithVignette(const cv::Mat &src, cv::Mat &dst);
void oilPainting(cv::Mat &inputImage, cv::Mat &outputImage);
void pencilSketch(cv::Mat &inputImage, cv::Mat &outputImage);



#endif
