
#include <opencv2/opencv.hpp>
/*
 * A function that detects and draws a square box around the markers.
 */
int detect_markers(cv::Mat &imageCopy, std::vector<std::vector<cv::Point2f>> &markerCorners,
				   std::vector<int> &markerIds);

/*
 * A function to draw three-d axis around the markers for each of the aruco-markers.
 */
int draw_3d_axes(std::vector<std::vector<cv::Point2f>> &markerCorners,
				 std::vector<int> markerIds,
				 cv::Mat &imageCopy,
				 cv::Mat &objPoints,
				 cv::Mat cameraMatrix,
				 cv::Mat distCoeffs);