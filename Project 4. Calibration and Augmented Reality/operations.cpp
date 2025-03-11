#include<iostream>
#include<opencv2/opencv.hpp>
#include "operations.h"

/*
 * A function that returns the world coordinates of a chess board with
   the number of internal corners.
 * Arg-1: point-set -> A opencv 3D Point container to store
          world coordinates.
 * Arg-2: rows -> The no.of rows in the internal corners.
 * Arg-3: cols -> The no.of cols in the internal corners.
 */
int get_world_coordinates(std::vector<cv::Vec3f> &point_set, int rows, int cols, float square_size) {
  // Iterate through rows.
  for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
	  point_set.emplace_back(j, -i, 0);
	}
  }
  return 0;
}

// counts and returns the total no of corner points in the container.
int get_corners_count(std::vector<std::vector<cv::Point2f>> &corners_list) {
  int corners_count = 0;
  for (const auto &i : corners_list) {
	for (int j = 0; j < i.size(); j++) {
	  corners_count++;
	}
  }
  return corners_count;
}

// counts and returns the total number of world coordinates in the container.
int get_points_count(std::vector<std::vector<cv::Vec3f>> &point_list) {
  int points_count = 0;
  // get the number of world_coordinates.
  for (const auto &i : point_list) {
	for (int j = 0; j < i.size(); j++) {
	  points_count++;
	}
  }
  return points_count;
}

/*
 * A function that tests whether the count of corners
   and their relative world coordinates are same.
 * Args-1: Pointlist -> A 2D vector of world coordinates.
 * Args-2: corners_list -> A 2D vector of corner points.

 Returns 0 if the count is same else returns 1.
 */
int check_validity(std::vector<std::vector<cv::Vec3f>> &point_list,
				   std::vector<std::vector<cv::Point2f>> &corners_list) {
  int corners_count = get_corners_count(corners_list);
  int points_count = get_points_count(point_list);

  if (corners_count==points_count)
	return 0;
  else
	return 1;
}

/*
 * A function that saves a given frame to local machine based on file_path.
 */
int save_frame(cv::Mat &frame) {
  std::string
	  file_path = "/Users/jyothivishnuvardhankolla/Desktop/Project-4-Calibration-Augmented-Reality/calibration_frames/";
  std::string fileFormat = ".jpeg";
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto timestamp = now_ms.time_since_epoch().count();
  std::string fileName = file_path + std::to_string(timestamp) + fileFormat;
  std::cout << "saving";
  cv::imwrite(fileName, frame);
  return 0;
}

/*
 * A function that initializes the camera and distortion matrix.
   Args-1:cameraMatrix    : An empty CV_64F mat object.
   Args-2:distortion_coeff: An empty CV_64F mat object.
 */

int initialize_camera_distortion_mats(cv::Mat &cameraMatrix, cv::Mat &distortion_coeff, cv::Mat &frame) {
  cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
  cameraMatrix.at<double>(0, 2) = frame.cols/2;
  cameraMatrix.at<double>(1, 2) = frame.rows/2;
  cameraMatrix.at<double>(0, 0) = 1;
  cameraMatrix.at<double>(1, 1) = 1;
  cameraMatrix.at<double>(2, 2) = 1;
  distortion_coeff = cv::Mat::zeros(8, 1, CV_64F);

  return 0;
}
/*
 * A function that performs camera calibration and return the reprojection error.
   Args-1:pont_list              : A 2d Vector of world-coordinates.
   Args-2:corners_list           : A 2d Vector of corners.
   Args-3:cameraMatrix           : A 3X3 CV_64F Mat object.
   Args-4:distortion_coefficients: A 3X3 CV_64F Mat object.
   Args-5:rotation_vector        : empty cv Mat object.
   Args-6:translation_vector     : empty cv Mat object.
 */
int perform_calibration(std::vector<std::vector<cv::Vec3f>> &point_list,
						std::vector<std::vector<cv::Point2f>> &corners_list,
						cv::Mat &cameraMatrix,
						cv::Mat &distortion_coefficients,
						std::vector<cv::Mat> &rotation_vector,
						std::vector<cv::Mat> &translation_vector,
						cv::Mat &frame) {

  std::cout << "Camera Matrix before calibration" << std::endl;
  std::cout << cameraMatrix << " " << std::endl;
  std::cout << "Distortion Coefficients before calibration" << std::endl;
  std::cout << distortion_coefficients << " " << std::endl;

  // call the calibrate function.
  double rms = cv::calibrateCamera(point_list,
								   corners_list,
								   frame.size(),
								   cameraMatrix,
								   distortion_coefficients,
								   rotation_vector,
								   translation_vector,
								   cv::CALIB_FIX_ASPECT_RATIO);
  std::cout << "Camera Matrix After calibration" << std::endl;
  std::cout << cameraMatrix << " " << std::endl;
  std::cout << "Distortion Coefficients After calibration" << std::endl;
  std::cout << distortion_coefficients << " " << std::endl;
  std::cout << "Rotation matrices" << std::endl;
  for (const auto &i : rotation_vector)
	std::cout << i << " " << std::endl;
  std::cout << "Translation matrices" << std::endl;
  for (const auto &i : translation_vector)
	std::cout << i << " " << std::endl;
  std::cout << "Reprojection erros" << std::endl;
  std::cout << rms << " " << std::endl;
  return 0;
}

/*
 * A function that saves cameraMatrix and distortion_coefficients into a XML file.
 */
int save_calibration(cv::Mat &cameraMatrix, cv::Mat &distortion_coefficients) {
  cv::FileStorage fs
	  ("/Users/jyothivishnuvardhankolla/Desktop/Project-4-Calibration-Augmented-Reality/camera_params.xml",
	   cv::FileStorage::WRITE);

  // Write the matrices to the file
  fs << "camera_matrix" << cameraMatrix;
  fs << "dist_coeffs" << distortion_coefficients;

  // Release the file storage
  fs.release();

  return 0;
}

/*
 * A function that displays cameraMatrix and distortionMatrix on the frame on real time.
 */
int display_rot_trans(cv::Mat &cameraMatrix, cv::Mat &distortionMatrix, cv::Mat &frame) {
  std::stringstream ss, ss1;
  ss << "Camera Matrix" << cameraMatrix;
  ss1 << "Distortion coefficients" << distortionMatrix;
  cv::putText(frame, ss.str(), cv::Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
  cv::putText(frame, ss1.str(), cv::Point(20, 80), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(234, 0, 255), 2);
  return 0;
}

/*
 * A function that projects a square object on to world-coordinates.
 */
int draw_square(cv::Mat &rotation_vector, cv::Mat &translation_vector, cv::Mat &cameraMatrix,
				cv::Mat &distortion_coefficient, cv::Mat &frame) {
  std::vector<cv::Point3f> objectPoints; // Vector to store world coordinates of object.
  // Points on 2d Plane.
  objectPoints.push_back(cv::Point3f(3, -1, 0));
  objectPoints.push_back(cv::Point3f(4, -1, 0));
  objectPoints.push_back(cv::Point3f(4, -2, 0));
  objectPoints.push_back(cv::Point3f(3, -2, 0));

  // Points on 3D Plane.
  objectPoints.push_back(cv::Point3f(3, -1, 5));
  objectPoints.push_back(cv::Point3f(4, -1, 5));
  objectPoints.push_back(cv::Point3f(4, -2, 5));
  objectPoints.push_back(cv::Point3f(3, -2, 5));

  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(objectPoints,
					rotation_vector,
					translation_vector,
					cameraMatrix,
					distortion_coefficient,
					imagePoints);

  // Bottom part of the cube.
  cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[1], imagePoints[2], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[3], imagePoints[0], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

  // Top part of the cube.
  cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[5], imagePoints[6], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[6], imagePoints[7], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[7], imagePoints[4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

  // lines connectiong bottom and top part of the cube.
  cv::line(frame, imagePoints[0], imagePoints[4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[1], imagePoints[5], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[2], imagePoints[6], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(frame, imagePoints[3], imagePoints[7], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  return 0;
}

/*
 * A function that draws and displays a cuboid on the world.
 */
int draw_house(cv::Mat &rotation_vector, cv::Mat &translation_vector, cv::Mat &cameraMatrix,
			   cv::Mat &distortion_coefficient, cv::Mat &frame, int darken) {
  std::vector<cv::Point3f> objectPoints;
  // make all pixels in the frame zero.
  if (darken==1) {
	for (int i = 0; i < frame.rows; i++) {
	  auto *rptr = frame.ptr<cv::Vec3b>(i);
	  for (int j = 0; j < frame.cols; j++) {
		for (int c = 0; c < 3; c++) {
		  rptr[j][c] = 0;
		}
	  }
	}
  }

  // Points for floor of the house.
  objectPoints.push_back(cv::Point3f(0, 0, 0));
  objectPoints.push_back(cv::Point3f(8, 0, 0));
  objectPoints.push_back(cv::Point3f(8, -5, 0));
  objectPoints.push_back(cv::Point3f(0, -5, 0));

  // points for roof of the house.
  objectPoints.push_back(cv::Point3f(0, 0, 5));
  objectPoints.push_back(cv::Point3f(8, 0, 5));
  objectPoints.push_back(cv::Point3f(8, -5, 5));
  objectPoints.push_back(cv::Point3f(0, -5, 5));

  // centre of the dome.
  objectPoints.push_back(cv::Point3f(4, -2, 9));

  // points for the door of the house.
  objectPoints.push_back(cv::Point3f(3, -5, 0));
  objectPoints.push_back(cv::Point3f(4, -5, 0));
  objectPoints.push_back(cv::Point3f(3, -5, 2.5));
  objectPoints.push_back(cv::Point3f(4, -5, 2.5));

  // Project the vertices onto the image plane
  std::vector<cv::Point2f> imagePoints;
  projectPoints(objectPoints, rotation_vector, translation_vector, cameraMatrix, distortion_coefficient, imagePoints);

  // constructiong the floor of the house.
  cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 5);
  cv::line(frame, imagePoints[1], imagePoints[2], cv::Scalar(0, 0, 255), 5);
  cv::line(frame, imagePoints[2], imagePoints[3], cv::Scalar(0, 0, 255), 5);
  cv::line(frame, imagePoints[3], imagePoints[0], cv::Scalar(0, 0, 255), 5);

  // constructiong the roof of the house.
  cv::line(frame, imagePoints[4], imagePoints[5], cv::Scalar(34, 124, 255), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[5], imagePoints[6], cv::Scalar(34, 124, 255), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[6], imagePoints[7], cv::Scalar(34, 124, 255), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[7], imagePoints[4], cv::Scalar(34, 124, 255), 5, cv::LINE_AA);

  // connecting floor and roof with pillars.
  cv::line(frame, imagePoints[0], imagePoints[4], cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[1], imagePoints[5], cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[2], imagePoints[6], cv::Scalar(0, 255, 0), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[3], imagePoints[7], cv::Scalar(0, 255, 0), 5, cv::LINE_AA);

  // joining roof with centre of the dome.
  cv::line(frame, imagePoints[4], imagePoints[8], cv::Scalar(255, 0, 0), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[5], imagePoints[8], cv::Scalar(255, 0, 0), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[6], imagePoints[8], cv::Scalar(255, 0, 0), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[7], imagePoints[8], cv::Scalar(255, 0, 0), 5, cv::LINE_AA);

  // constructing the door for the house.
  cv::line(frame, imagePoints[9], imagePoints[11], cv::Scalar(255, 0, 255), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[10], imagePoints[12], cv::Scalar(255, 0, 255), 5, cv::LINE_AA);
  cv::line(frame, imagePoints[11], imagePoints[12], cv::Scalar(255, 0, 255), 5, cv::LINE_AA);

  return 0;
}

/*
 * A functions that detects the Harris corners in the frame of a video.
 */
int detect_harris_corners(cv::Mat &grayFrame, cv::Mat &frame) {
  cv::Mat destImg, destImgnorm, destImgnorm_scaled;
  cv::cornerHarris(grayFrame, destImg, 2, 3, 0.04, cv::BORDER_DEFAULT);
  cv::normalize(destImg, destImgnorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(destImgnorm, destImgnorm_scaled);

  for (int i = 0; i < destImgnorm.rows; i++) {
	for (int j = 0; j < destImgnorm.cols; j++) {
	  if ((int)destImgnorm.at<float>(i, j) > 180) {
		cv::circle(frame, cv::Point(j, i), 10, cv::Scalar(0, 0, 255), 2, 8, 0);
	  }
	}
  }
  return 0;
}