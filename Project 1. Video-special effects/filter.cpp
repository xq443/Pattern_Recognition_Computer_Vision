/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1 task 4
 */
#include "filter.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Function to apply grayscale transformation to a frame
void applyGrayscale(cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "Error: Empty frame provided to applyGrayscale function.\n";
        return;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
}

int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Check if the source image has 3 channels (RGB)
    if (src.channels() != 3) {
        std::cerr << "Input image must have 3 channels (RGB).\n";
        return -1; // Error
    }

    dst = src;

    // Loop through each pixel
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            // Access the pixel at (row, col)
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);

            // Perform the creative transformation on each channel
            uchar r = 255 - pixel[2]; // Red channel, subtract from 255
            uchar g = 255 - pixel[1]; // Green channel, subtract from 255
            uchar b = 255 - pixel[0]; // Blue channel, subtract from 255

            // Set all three channels of the destination pixel to the same value
            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(b, g, r);
        }
    }

    return 0; // Success
}

void applySepiaTone(cv::Mat &src, cv::Mat &dst) {
    dst = src;

    // Loop through each pixel
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            // Access the pixel at (row, col)
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);

            // Original RGB values
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            // Sepia transformation with clamping
            uchar newB = std::min(255.0, 0.272 * R + 0.534 * G + 0.131 * B);
            uchar newG = std::min(255.0, 0.349 * R + 0.686 * G + 0.168 * B);
            uchar newR = std::min(255.0, 0.393 * R + 0.769 * G + 0.189 * B);

            // Assign the new values to the destination image
            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(newB, newG, newR);
        }
    }
}


// Function to apply a 3x3 Sobel X filter
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    // Check if the source image has 3 channels (RGB)
    if (src.channels() != 3) {
        std::cerr << "Input image must have 3 channels (RGB).\n";
        return -1; // Error
    }

    // Initialize the destination image
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Define the Sobel X kernel
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Apply the Sobel X filter
    for (int row = 1; row < src.rows - 1; ++row) {
        for (int col = 1; col < src.cols - 1; ++col) {
            cv::Vec3s sum = cv::Vec3s(0, 0, 0);
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(row + i, col + j);
                    for (int k = 0; k < 3; ++k) {
                        sum[k] += pixel[k] * kernelX[i + 1][j + 1];
                    }
                }
            }
            dst.at<cv::Vec3s>(row, col) = sum;
        }
    }

    return 0; // Success
}

// Function to apply a 3x3 Sobel Y filter
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    // Check if the source image has 3 channels (RGB)
    if (src.channels() != 3) {
        std::cerr << "Input image must have 3 channels (RGB).\n";
        return -1; // Error
    }

    // Initialize the destination image
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Define the Sobel Y kernel
    int kernelY[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

    // Apply the Sobel Y filter
    for (int row = 1; row < src.rows - 1; ++row) {
        for (int col = 1; col < src.cols - 1; ++col) {
            cv::Vec3s sum = cv::Vec3s(0, 0, 0);
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(row + i, col + j);
                    for (int k = 0; k < 3; ++k) {
                        sum[k] += pixel[k] * kernelY[i + 1][j + 1];
                    }
                }
            }
            dst.at<cv::Vec3s>(row, col) = sum;
        }
    }

    return 0; // Success
}

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Check if 3-channel signed short images
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3 || sx.size() != sy.size()) {
        std::cerr << "Input images must be of type CV_16SC3 and have the same size.\n";
        return -1; // Error
    }

    // Initialize the destination image
    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    // Loop through each pixel
    for (int row = 0; row < sx.rows; ++row) {
        for (int col = 0; col < sx.cols; ++col) {
            // Access the pixels at (row, col)
            cv::Vec3s pixelSx = sx.at<cv::Vec3s>(row, col);
            cv::Vec3s pixelSy = sy.at<cv::Vec3s>(row, col);
            cv::Vec3b pixelDst;

            // Calculate the gradient magnitude for each channel
            for (int k = 0; k < 3; ++k) {
                float magnitude = std::sqrt(pixelSx[k] * pixelSx[k] + pixelSy[k] * pixelSy[k]);
                pixelDst[k] = cv::saturate_cast<uchar>(magnitude);
            }

            // Assign the calculated magnitude to the destination image
            dst.at<cv::Vec3b>(row, col) = pixelDst;
        }
    }

    return 0; // Success
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    // Check if the source image has 3 channels (RGB)
    if (src.channels() != 3) {
        std::cerr << "Input image must have 3 channels (RGB).\n";
        return -1; // Error
    }

    // Blur the image
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(15, 15), 0); // Apply a Gaussian blur

    // Calculate the size of each bucket
    float b = 255.0 / levels;

    // Initialize the destination image
    dst = blurred;

    // Loop through each pixel
    for (int row = 0; row < blurred.rows; ++row) {
        for (int col = 0; col < blurred.cols; ++col) {
            // Access the pixel at (row, col)
            cv::Vec3b pixel = blurred.at<cv::Vec3b>(row, col);

            // Quantize each channel
            for (int k = 0; k < 3; ++k) {
                float x = pixel[k];
                float xt = std::round(x / b);
                float xf = xt * b;
                dst.at<cv::Vec3b>(row, col)[k] = cv::saturate_cast<uchar>(xf); // convert the xf into a range of 0 to 255.
            }
        }
    }

    return 0; // Success
}