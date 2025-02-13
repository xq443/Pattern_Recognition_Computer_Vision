/*
  Xujia Qin 12th Feb, 2025
  S21
*/

#ifndef FILTERS_H
#define FILTERS_H

/*
 * Function to implement Thresholding.
 * src: Source HSV Image on which the Thresholding need to be applied.
 * dst: Destination container to store the Image after Applying Thresholding.
 */
int threshold(cv::Mat &src, cv::Mat &dst);

/*
 * Function to perform 4-connected Grassfire Transform given HSV thresholded Image.
 * src: Thresholded HSV Image.
 * Return a 2d vector where each cell contains distance to nearest background pixel.
 */
std::vector<std::vector<int>> GrassfireTransform(cv::Mat &src);

/*
 * Function to implement 8-connected Grassfire Transform.
 * src: Thresholded HSV Image.
 * Return a 2d vector where each cell contains distance to nearest foreground pixel.
 */
std::vector<std::vector<int>> GrassfireTransform1(cv::Mat &src);

/*
 * Funtion to perform Erosion.
 * Arg1-distances: distances matrix after performing Grassfire transform.
 * Arg-2 erosion_length: Number of erosions to perform.
 */
int Erosion(std::vector<std::vector<int>> &distances, cv::Mat &src, int erosion_length);

/*
 * Funtion to perform Dialation.
 * Arg1-distances: distances matrix after performing Grassfire transform.
 * Arg-2-distances:
 * Arg-3 erosion_length: Number of dialtaions to perform.
 */
int Dialation(std::vector<std::vector<int>> &distances, cv::Mat &src, int dialation_length);

/*
 * Function to perform Segmentation.
 * Arg-1-src: Thresholded RGB image on which segmentation to be performed.
 * returns a new Image where the Top-N components are filled with random colors.
 */
cv::Mat SegmentImage(cv::Mat &src);

/*
 * Function to compute the moments of regions in a given image.
 * Args-1-src  : Thrsholded RGB Image.
 * Returns a new Image with central Axis drawn on it.
 */
cv::Mat calculate_moments(cv::Mat &src);

std::vector<double> get_moments(cv::Mat &src, char thresh_type[]);
/*
 * Function that stores Moments as fearues in a csv file given a Thresholded RGB Image.
 */
int collect_data(cv::Mat &src, char threshtype[], std::string label = "");
#endif 
