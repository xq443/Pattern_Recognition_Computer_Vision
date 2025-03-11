#include <opencv2/opencv.hpp>
/*
 * A function that returns the world coordinates of a chess board with
   the number of internal corners.
 * Arg-1: point-set -> A opencv 3D Point container to store
          world coordinates.
 * Arg-2: rows -> The no.of rows in the internal corners.
 * Arg-3: cols -> The no.of cols in the internal corners.
 */
int get_world_coordinates(std::vector<cv::Vec3f> &point_set, int rows, int cols, float square_size);

/*
 * A function that tests whether the count of corners
   and their relative world coordinates are same.
 * Args-1: Pointlist -> A 2D vector of world coordinates.
 * Args-2: corners_list -> A 2D vector of corner points.

 Returns 0 if the count is same else returns 1.
 */
int check_validity(std::vector<std::vector<cv::Vec3f>> &point_list,
				   std::vector<std::vector<cv::Point2f>> &corners_list);

/*
 * A function that saves a given frame to local machine based on file_path.
 */
int save_frame(cv::Mat &frame);

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
						cv::Mat &frame);

/*
 * A function that initializes the camera and distortion matrix.
   Args-1:cameraMatrix    : An empty CV_64F mat object.
   Args-2:distortion_coeff: An empty CV_64F mat object.
 */
int initialize_camera_distortion_mats(cv::Mat &cameraMatrix, cv::Mat &distortion_coeff, cv::Mat &frame);

/*
 * A function that saves cameraMatrix and distortion_coefficients into a XML file.
 */
int save_calibration(cv::Mat &cameraMatrix, cv::Mat &distortion_coefficients);

/*
 * A function that displays cameraMatrix and distortionMatrix on the frame on real time.
 */
int display_rot_trans(cv::Mat &cameraMatrix, cv::Mat &distortionMatrix, cv::Mat &frame);

/*
 * A function that projects a square object on to world-coordinates.
 */
int draw_square(cv::Mat &rotation_vector, cv::Mat &translation_vector, cv::Mat &cameraMatrix,
				cv::Mat &distortion_coefficient, cv::Mat &frame);

/*
 * A function that draws and displays a cuboid on the world.
 */
int draw_house(cv::Mat &rotation_vector, cv::Mat &translation_vector, cv::Mat &cameraMatrix,
			   cv::Mat &distortion_coefficient, cv::Mat &frame, int darken);

/*
 * A functions that detects the Harris corners in the frame of a video.
 */
int detect_harris_corners(cv::Mat &grayFrame, cv::Mat &frame);