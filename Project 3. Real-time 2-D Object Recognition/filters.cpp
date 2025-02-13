/*
  Xujia Qin 12th Feb, 2025
  S21
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include "filters.h"
#include "csv_utils.h"
#include <cmath>

using namespace std;

// comparator function to sort vector pair based on value.
bool cmp(pair<int, int> &a, pair<int, int> &b) {
  return a.second < b.second;
}

// function to fill pixels value in each hue, sat, val channels of hsv color space.
void fill_pixels(cv::Vec3b *rptr, int col, int h_value, int s_value, int v_value) {
  rptr[col][0] = h_value;
  rptr[col][1] = s_value;
  rptr[col][2] = v_value;
}

// helper function to get moment values given a thresholded RGB image.
vector<double> get_moments(cv::Mat &src, char threshtype[]) {
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // Apply adaptive thresholding
  cv::Mat binary;
  if (::strcmp(threshtype, "adaptive")==0)
	cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 2);
  else {
	binary = gray.clone();
  }

  vector<double> features;
  cv::Moments moments = cv::moments(binary, true);
  double huMoments[7];
  cv::HuMoments(moments, huMoments);
  for (int i = 0; i < 7; i++) {
	features.push_back(-1*copysign(1.0, huMoments[i])*log10(abs(huMoments[i])));
  }
  return features;
}

/*
 * Function to implement Thresholding.
 * src: Source HSV Image on which the Thresholding need to be applied.
 * dst: Destination container to store the Image after Applying Thresholding.
 */
int threshold(cv::Mat &src, cv::Mat &dst) {
  dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
  // Define the lower and upper boundaries for white color in HSV space.
  int hue_low, sat_low, val_low, hue_high, sat_high, val_high;
  hue_low = 0;
  sat_low = 0;
  val_low = 180;

  hue_high = 250;
  sat_high = 30;
  val_high = 255;

  // Iterate through rows.
  for (int i = 0; i < src.rows; i++) {
	// create row pointers to access indexes.
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
	// Iterate through columns.
	for (int j = 0; j < src.cols; j++) {
	  int hue = rptr[j][0];
	  int sat = rptr[j][1];
	  int val = rptr[j][2];
	  if ((hue >= hue_low && hue <= hue_high) && (sat >= sat_low && sat <= sat_high)
		  && (val >= val_low && val <= val_high)) {
		fill_pixels(dptr, j, 0, 0, 0);
	  } else {
		if (sat < 150) {
		  fill_pixels(dptr, j, 179, 25, 255);
		} else {
		  fill_pixels(dptr, j, 179, 25, 235);
		}
	  }
	}
  }
  return 0;
}