
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>

using namespace cv;
using namespace std;

cv::Mat calculate_moments(cv::Mat &src) {
  cv::Mat Thresholded_Grayscale_img, central_moment_image;
  central_moment_image = src.clone();
  cv::cvtColor(src, Thresholded_Grayscale_img, cv::COLOR_BGR2GRAY);

  // calculate the moments of the Image.
  cv::Moments moments = cv::moments(Thresholded_Grayscale_img);

  // caluclate the Hu moments of the Image.
  cv::Mat hu_moments;
  cv::HuMoments(moments, hu_moments);

  // calculate the axis of central moments.
  double u20 = hu_moments.at<double>(2, 0);
  double u02 = hu_moments.at<double>(0, 2);
  double u11 = hu_moments.at<double>(1, 1);
  double theta = 0.5*::atan2(2*u11, u20 - u02);

  // calculate centroid of the image.
  double x_cor = moments.m10/moments.m00;
  double y_cor = moments.m01/moments.m00;

  // calculate the endpoints of the line for the axis of central moments.
  double cos_theta = cos(theta);
  double sin_theta = sin(theta);

  double x1_point = x_cor - 400*sin_theta;
  double y1_point = y_cor - 400*cos_theta;
  double x2_point = x_cor + 400*sin_theta;
  double y2_point = y_cor + 400*cos_theta;

  double angle = 0.5*(::atan((2*u11)/(u20 - u02)));


  // Draw axis of central moment on the Image.
  cv::line(central_moment_image,
		   cv::Point(x1_point, y1_point),
		   cv::Point(x2_point, y2_point),
		   cv::Scalar(0, 0, 255),
		   5);

  // draw a oriented bounding box.
  cv::Point centerPoint(x_cor, y_cor);
  cv::Scalar color = cv::Scalar(234, 234, 123);

  // find the size of rectangle.
  cv::Mat ImageIds, stats_matrix, centroids;
  int num_components =
	  cv::connectedComponentsWithStats(Thresholded_Grayscale_img, ImageIds, stats_matrix, centroids, 4);
  double height = stats_matrix.at<int>(2, cv::CC_STAT_HEIGHT);
  double width = stats_matrix.at<int>(2, cv::CC_STAT_HEIGHT);

  cout << height << " " << width << endl;

  // create a rotated rectangle.
  cv::RotatedRect rotatedRectangle(centerPoint, cv::Size2f(width, height), angle);

  // We take the edges that OpenCV calculated for us
  cv::Point2f vertices2f[4];
  rotatedRectangle.points(vertices2f);

  // Convert them so we can use them in a fillConvexPoly
  for (int i = 0; i < 4; ++i) {
	cv::line(central_moment_image, vertices2f[i], vertices2f[(i + 1)%4], cv::Scalar(0, 255, 0), 5);
  }

  // Now we can fill the rotated rectangle with our specified color
  cv::Rect brect = rotatedRectangle.boundingRect();
  return central_moment_image;
}

int main() {
  // Load an image
  std::string imagePath = "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 3. Real-time 2-D Object Recognition/db/f4.png";  // Change this to your image file path
  cv::Mat src = cv::imread(imagePath, cv::IMREAD_COLOR);

  if (src.empty()) {
    std::cerr << "Error: Could not load image." << std::endl;
    return -1;
  }

  // Call the calculate_moments function
  cv::Mat result = calculate_moments(src);

  // Display the result
  cv::imshow("Moment Image", result);

  // Wait for any key press and then close the image window
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}
