//
// Created by Jyothi vishnu vardhan Kolla on 2/4/23.
// CS-5330 Spring semester.
//
#include <opencv2/opencv.hpp>
/*
 * Function to implement 3 X 3 Sobel X filter
 * src: Source Image
 * dst: Destination Image, allocated by this function
 */
int sobelX3X3(cv::Mat &src, cv::Mat &dst);
/*
 * Function to implement 3 X 3 Sobel y filter
 * src: Source Image
 * dst: Destination Image, allocated by this function
 */
int sobelY3X3(cv::Mat &src, cv::Mat &dst);
/*
 * Function to generate a gradient magnitude image from the X and Y sobel Images.
 * sx: Short sobelX Filter generated Image.
 * sy: Short sobely Filter generated Image.
 * dst: Destination Container where the Image will be stored.
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

/*
 * Function to do Laplacian Filter on an given Image.
 * [ 0   1  0 ]
 * [ -1  4   1]
 * [  0  -1   0]
 * src: Source Image on which laplacian Filter must be applied.
 * dst: Destination container to store the Filtered Image.
 */
int laplacianFilter(cv::Mat &src, cv::Mat &dst);