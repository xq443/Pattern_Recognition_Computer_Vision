/*
  Xujia Qin 
  30th Jan, 2025
  S21
*/


#include <iostream>
#include "utils.h"
#include <opencv2/opencv.hpp>
using namespace std;

/*
 * Function to implement 3 X 3 Sobel X filter
 * src: Source Image
 * dst: Destination Image, allocated by this function
 */
int sobelX3X3(cv::Mat &src, cv::Mat &dst) {
  // fill the destination Mat with zeros.
  dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);
  cv::Mat temp = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

  // 1. convolving with 1 X 3 filter
  // loop through rows.
  for (int i = 1; i < src.rows - 1; i++) {
	// pointers pointing to rows of src Image.
	cv::Vec3b *srcptrm1 = src.ptr<cv::Vec3b>(i - 1);
	cv::Vec3b *srcptr = src.ptr<cv::Vec3b>(i);
	cv::Vec3b *srcptrp1 = src.ptr<cv::Vec3b>(i + 1);

	// Pointer pointing to destination Image.
	cv::Vec3s *tempptr = temp.ptr<cv::Vec3s>(i);
	// loop through columns.
	for (int j = 1; j < src.cols - 1; j++) {
	  // loop through channels
	  for (int c = 0; c < 3; c++) {
		tempptr[j][c] = (1*srcptrm1[j][c] + 2*srcptr[j][c] + 1*srcptrp1[j][c])/4;
	  }
	}
  }

  // 2. convolving with 3 X 1 filter
  // loop through rows.
  for (int i = 1; i < src.rows - 1; i++) {
	// pointers pointing to rows of src Image.
	cv::Vec3s *tempptr = temp.ptr<cv::Vec3s>(i);

	// Pointer pointing to destination Image.
	cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);
	// loop through columns.
	for (int j = 1; j < src.cols - 1; j++) {
	  // loop through channels
	  for (int c = 0; c < 3; c++) {
		dstptr[j][c] = (-1*tempptr[j - 1][c] + 1*tempptr[j + 1][c]);
	  }
	}
  }
  return (0);
}

/*
 * Function to implement 3 X 3 Sobel X filter
 * src: Source Image
 * dst: Destination Image, allocated by this function
 */
int sobelY3X3(cv::Mat &src, cv::Mat &dst) {
  // fill the destination Mat with zeros.
  dst = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);
  cv::Mat temp = cv::Mat::zeros(src.rows, src.cols, CV_16SC3);

  // 1. convolving with 1 X 3 filter
  // loop through rows.
  for (int i = 1; i < src.rows - 1; i++) {
	// pointers pointing to rows of src Image.
	cv::Vec3b *srcptrm1 = src.ptr<cv::Vec3b>(i - 1);
	cv::Vec3b *srcptr = src.ptr<cv::Vec3b>(i);
	cv::Vec3b *srcptrp1 = src.ptr<cv::Vec3b>(i + 1);

	// Pointer pointing to destination Image.
	cv::Vec3s *tempptr = temp.ptr<cv::Vec3s>(i);
	// loop through columns.
	for (int j = 1; j < src.cols - 1; j++) {
	  // loop through channels
	  for (int c = 0; c < 3; c++) {
		tempptr[j][c] = (1*srcptrm1[j][c] + -1*srcptrp1[j][c])/4;
	  }
	}
  }

  // 2. convolving with 3 X 1 filter
  // loop through rows.
  for (int i = 1; i < src.rows - 1; i++) {
	// pointers pointing to rows of src Image.
	cv::Vec3s *tempptr = temp.ptr<cv::Vec3s>(i);

	// Pointer pointing to destination Image.
	cv::Vec3s *dstptr = dst.ptr<cv::Vec3s>(i);
	// loop through columns.
	for (int j = 1; j < src.cols - 1; j++) {
	  // loop through channels
	  for (int c = 0; c < 3; c++) {
		dstptr[j][c] = (1*tempptr[j - 1][c] + 2*tempptr[j][c] + 1*tempptr[j + 1][c]);
	  }
	}
  }
  return (0);
}

/*
 * Function to generate a gradient magnitude image from the X and Y sobel Images.
 * sx: Short sobelX Filter generated Image.
 * sy: Short sobely Filter generated Image.
 * dst: Destination Container where the Image will be stored.
 */

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
  dst = cv::Mat::zeros(sx.rows, sx.cols, CV_8UC3);

  // loop through rows.
  for (int i = 0; i < sx.rows; i++) {
    // create row pointers for sx, sy, dst
    cv::Vec3s *sxrptr = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *syrptr = sy.ptr<cv::Vec3s>(i);
    cv::Vec3b *dstrptr = dst.ptr<cv::Vec3b>(i);

    // loop through columns.
    for (int j = 0; j < sx.cols; j++) {
      // loop through color channels.
      for (int c = 0; c < 3; c++) {
        // Calculate gradient magnitude
        float mag = sqrt((sxrptr[j][c] * sxrptr[j][c]) + (syrptr[j][c] * syrptr[j][c]));
        
        // Clamp the value to the 0-255 range
        dstrptr[j][c] = cv::saturate_cast<uchar>(mag);
      }
    }
  }
  return 0;
}



/*
 * Function to do Laplacian Filter on an given Image.
 * src: Source Image on which laplacian Filter must be applied.
 * dst: Destination container to store the Filtered Image.
 */
int laplacianFilter(cv::Mat &src, cv::Mat &dst) {
  // create a destination pointer with zeros
  dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);

  //cv::Mat greyscaleimg;
  //greyscale(src, greyscaleimg);

  // loop over rows.
  for (int i = 1; i < src.rows - 1; i++) {
	// row pointers.
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(i - 1);
	cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(i + 1);

	cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
	// loop over columns.
	for (int j = 1; j < src.cols - 1; j++) {
	  // loop over channels.
	  for (int c = 0; c < 3; c++) {
		dptr[j][c] = (4*rptr[j][c] + -1*rptr[j + 1][c] + -1*rptr[j - 1][c] + -1*rptrp1[j][c] + -1*rptrm1[j][c])/6;
	  }
	}
  }
  return (0);
}