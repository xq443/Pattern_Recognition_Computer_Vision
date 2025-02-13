/*
  Xujia Qin 12th Feb, 2025
  S21
*/

#include <iostream>
#include <experimental/filesystem>
#include<opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "filters.h"

using namespace std;
namespace fs = std::experimental::filesystem;

string get_label(string path, char delimeter) {
  vector<string> words;
  size_t first;
  size_t last = 0;

  while ((first = path.find_first_not_of(delimeter, last))
	  !=string::npos) {
	last = path.find(delimeter, first);
	words.push_back(path.substr(first, last - first));
  }
  return words[6];
}

void iterateDirectory(fs::path path, char thresh_type[]) {
  for (const auto &entry : fs::directory_iterator(path)) {
	if (entry.is_directory()) {
	  // get the path and name of the subdirectory
	  fs::path subdir_path = entry.path();
	  // recursively call iterateDirectory for the subdirectory
	  iterateDirectory(subdir_path, thresh_type);
	} else {
	  if (entry.path().filename()!=".DS_Store")
		string img_path =
			entry.path();
	  cv::Mat blurred_color_image, HSV_Image; // Mat object to store blurred, HSV_images.
	  cv::Mat color_image = cv::imread(entry.path()); // Mat object to store original frame.
	  if (color_image.empty()) {
		continue;
	  }
	  cv::medianBlur(color_image, blurred_color_image, 5); // Blurring the color Image.
	  cv::cvtColor(blurred_color_image, HSV_Image, cv::COLOR_BGR2HSV); // Turing into HSV color space.
	  cv::Mat HSVthresholded_image; // Mat object to store thresholded image.
	  threshold(HSV_Image, HSVthresholded_image); // Threshold the Hsv image.
	  vector<vector<int>>
		  Erosion_distance = GrassfireTransform(HSVthresholded_image); // Vector to store Erosion distances.
	  Erosion(Erosion_distance, HSVthresholded_image, 5); // Perfrom Erosion.
	  vector<vector<int>>
		  Dialation_distance = GrassfireTransform1(HSVthresholded_image); // Vector to store Dialation distances.
	  Dialation(Dialation_distance, HSVthresholded_image, 5); // Perform Dialation.
	  cv::Mat thresholded_Image; // mat object to store final thresholded RGB Image.
	  cv::cvtColor(HSVthresholded_image, thresholded_Image, cv::COLOR_HSV2BGR);
	  string label = get_label(entry.path(), '/');
	  if (::strcmp(thresh_type, "adaptive")==0) {
		collect_data(color_image, thresh_type, label);
	  } else {
		collect_data(thresholded_Image, thresh_type, label);
	  }
	}
  }
}

int main(int argc, char *argv[]) {
  fs::path path = "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 3. Real-time 2-D Object Recognition/db";
  char thresh_type[256];
  ::strcpy(thresh_type, argv[1]);
  cout << thresh_type;
  iterateDirectory(path, thresh_type);

  return 0;
}