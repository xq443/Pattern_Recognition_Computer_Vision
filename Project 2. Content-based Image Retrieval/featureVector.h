/*
  Xujia Qin 
  30th Jan, 2025
  S21
*/

#include <cstdio>
#include "opencv2/dnn.hpp"     // DNN API include file
#include <vector>

/*
 * Retrives the feature vector values given an image by applyig a
 * 7 * 7 square in the center of the image.
 */
std::vector<float> sevenXSevenSquare(cv::Mat &src);

/*
 * Computes the 2D-histogram for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
std::vector<float> twodHistogram(cv::Mat &src, int bins = 16);

/*
 * Computes a 3D-histogram for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
std::vector<float> ThreedHistogram(cv::Mat &src, int bins = 8);

// /*
//  * Computes a multi 3D-histogram for a given top and bottom half of image.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */
std::vector<float> multiHistogram(cv::Mat &src, int bins = 8);

// /*
//  * Computes a multi 3D-histogram for a given Image from left and right half of image.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */
// std::vector<float> multiHistogramLeftRight(cv::Mat &src, int bins = 8);

/*
 * Computes a multi 3D-histogram by taking the gradient magnitude for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
std::vector<float> colorTexture(cv::Mat &src);

// /*
//  * Computes a multi 3D-histogram for a given Image one for image and another by applying gradient magnitude on it.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */
std::vector<float> LaplaciancolorTexture(cv::Mat &src);
std::vector<float> extractFeatureVector(const char *targetFilename, const char *featuresFile);

// /*
//  * Thresholds the given Image in HSV format in such a way that, all yellow
//  * pixels are whitened.
//  */
std::vector<float> yellowThresholding(cv::Mat &src);

std::vector<float> depthFilteredMultiHistogram(cv::Mat &src, cv::Mat &depthMap, int bins, float depthThreshold);

cv::Mat computeDepthMap(cv::Mat &src);
std::vector<float> extractHOGFeatures(cv::Mat &src);
std::vector<float> openCVEmbedding(cv::Mat &src, int debug);