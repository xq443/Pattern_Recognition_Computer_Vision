/*
  Xujia Qin
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/
#include "timeBlur.h"
#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

// prototypes for the functions to test
int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );

// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}


int blur5x5_1( cv::Mat &src, cv::Mat &dst ) {
  dst = src.clone();

  // Gaussian
  int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
  };

  int sum = 100;

  // Loop through the image, skipping the outer two rows and columns
    for (int row = 2; row < src.rows - 2; ++row) {
        for (int col = 2; col < src.cols - 2; ++col) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the 5x5 kernel
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(row + i, col + j);
                    int weight = kernel[i + 2][j + 2];
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Normalize and clamp the values to the 0-255 range
            uchar newB = std::min(255, std::max(0, sumB / sum));
            uchar newG = std::min(255, std::max(0, sumG / sum));
            uchar newR = std::min(255, std::max(0, sumR / sum));

            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(newB, newG, newR);
        }
    }

    return 0; // success
}


int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1; // Return an error code for invalid input
    }

    dst = src.clone(); // Create a deep copy of the source image
    
    // Separable kernel (1D)
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10;

    // Temporary buffer for intermediate horizontal pass
    cv::Mat temp = src.clone();

    // Horizontal pass (row-wise)
    for (int row = 0; row < src.rows; ++row) {
        cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(row);
        cv::Vec3b* tempRow = temp.ptr<cv::Vec3b>(row);

        for (int col = 2; col < src.cols - 2; ++col) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Convolve the kernel horizontally
            for (int k = -2; k <= 2; ++k) {
                cv::Vec3b pixel = srcRow[col + k];
                int weight = kernel[k + 2];
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            tempRow[col] = cv::Vec3b(
                std::min(255, std::max(0, sumB / kernelSum)),
                std::min(255, std::max(0, sumG / kernelSum)),
                std::min(255, std::max(0, sumR / kernelSum))
            );
        }
    }

    // Vertical pass (column-wise)
    for (int col = 0; col < src.cols; ++col) {
        for (int row = 2; row < src.rows - 2; ++row) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Convolve the kernel vertically
            for (int k = -2; k <= 2; ++k) {
                cv::Vec3b pixel = temp.at<cv::Vec3b>(row + k, col); // Access vertically
                int weight = kernel[k + 2];
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(
                std::min(255, std::max(0, sumB / kernelSum)),
                std::min(255, std::max(0, sumG / kernelSum)),
                std::min(255, std::max(0, sumR / kernelSum))
            );
        }
    }

    return 0; // Success
}