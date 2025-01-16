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
