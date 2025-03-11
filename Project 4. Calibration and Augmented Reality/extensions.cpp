#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/aruco.hpp>
#include "extensions.h"

/*
 * A function that detects and draws a square box around the markers.
 */
int detect_markers(cv::Mat &imageCopy, std::vector<std::vector<cv::Point2f>> &markerCorners,
				   std::vector<int> &markerIds) {
  if (!markerIds.empty())
	cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds, cv::Scalar(0, 255, 0));
  return 0;
}

/*
 * A function to draw three-d axis around the markers for each of the aruco-markers.
 */
int draw_3d_axes(std::vector<std::vector<cv::Point2f>> &markerCorners,
				 std::vector<int> markerIds,
				 cv::Mat &imageCopy,
				 cv::Mat &objPoints,
				 cv::Mat cameraMatrix,
				 cv::Mat distCoeffs) {
  int nMarkers = markerCorners.size();
  std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

  // calculate Pose for each marker.
  for (int i = 0; i < nMarkers; i++) {
	cv::solvePnP(objPoints, markerCorners[i], cameraMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));
  }

  // Draw axis for each marker.
  for (unsigned int i = 0; i < markerIds.size(); i++) {
	cv::drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
  }
  return 0;
}