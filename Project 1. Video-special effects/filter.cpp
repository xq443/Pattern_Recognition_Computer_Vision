/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1 task 4
  Note: for further optimization, I can use ptr to access pixel instead of at method
 */
#include "filter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>  
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
    dst = src;
    for (int row = 0; row < dst.rows; ++row) {
        for (int col = 0; col < dst.cols; ++col) {
            cv::Vec3b pixel = dst.at<cv::Vec3b>(row, col);

            // Calculate the grayscale value using the formula
            uchar gray = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);

            // Set all three channels of the destination pixel to the grayscale value
            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(gray, gray, gray);
        }
    }

    return 0; // success
}

void applySepiaTone(cv::Mat &frame) {
    if (frame.empty()) {
        std::cerr << "Error: Empty frame provided to applyGrayscale function.\n";
        return;
    }

    for (int row = 0; row < frame.rows; ++row) {
        for (int col = 0; col < frame.cols; ++col) {
            cv::Vec3b pixel = frame.at<cv::Vec3b>(row, col);

            // RGB values
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            // Sepia transformation with clamping
            uchar newB = std::min(255, static_cast<int>(0.272 * R + 0.534 * G + 0.131 * B));
            uchar newG = std::min(255, static_cast<int>(0.349 * R + 0.686 * G + 0.168 * B));
            uchar newR = std::min(255, static_cast<int>(0.393 * R + 0.769 * G + 0.189 * B));

            // Assign the new values to the destination image
            frame.at<cv::Vec3b>(row, col) = cv::Vec3b(newB, newG, newR);
        }
    }
}

void applySepiaToneWithVignette(const cv::Mat &src, cv::Mat &dst) {
    dst = src;
    int rows = src.rows;
    int cols = src.cols;

    // Calculate the center of the image
    cv::Point center(cols / 2, rows / 2);

    // Calculate the max distance from the center
    double maxDist = std::sqrt(center.x * center.x + center.y * center.y);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);

            // RGB values
            uchar B = pixel[0];
            uchar G = pixel[1];
            uchar R = pixel[2];

            // Sepia transformation with clamping
            uchar newB = std::min(255.0, 0.272 * R + 0.534 * G + 0.131 * B);
            uchar newG = std::min(255.0, 0.349 * R + 0.686 * G + 0.168 * B);
            uchar newR = std::min(255.0, 0.393 * R + 0.769 * G + 0.189 * B);

            // the distance of the pixel from the center
            double dist = std::sqrt(std::pow(col - center.x, 2) + std::pow(row - center.y, 2));

            // the vignette factor (closer to center = factor closer to 1)
            double vignetteFactor = 1.0 - (dist / maxDist); // Scale factor: 1 at center, 0 at edges

            // Apply the vignette effect
            vignetteFactor = std::max(0.0, vignetteFactor); // Ensure it's non-negative
            newB = static_cast<uchar>(newB * vignetteFactor);
            newG = static_cast<uchar>(newG * vignetteFactor);
            newR = static_cast<uchar>(newR * vignetteFactor);

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

    for (int row = 0; row < sx.rows; ++row) {
        for (int col = 0; col < sx.cols; ++col) {
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

    for (int row = 0; row < blurred.rows; ++row) {
        for (int col = 0; col < blurred.cols; ++col) {
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

// additional effects
void cartoonEffect(cv::Mat &src, cv::Mat &dst) {
    cv::Mat gray, edges, color;

    // Convert to grayscale
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply median blur to reduce noise
    cv::medianBlur(gray, gray, 7);

    // Detect edges using the Laplacian operator
    cv::Laplacian(gray, edges, CV_8U, 5);

    // Threshold the edges to create a binary mask
    cv::threshold(edges, edges, 80, 255, cv::THRESH_BINARY_INV);

    // Apply bilateral filter to smooth colors while preserving edges
    cv::bilateralFilter(src, color, 9, 150, 150);

    // Enhance edges by dilating the edge mask
    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

    // Combine the quantized colors with the enhanced edge mask
    cv::bitwise_and(color, color, dst, edges);
}

void embossEffect(cv::Mat &src, cv::Mat &dst) {
    // Convert source image to grayscale
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    // Sobel gradients
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_32F, 1, 0, 3); // Gradient in X direction
    cv::Sobel(gray, sobelY, CV_32F, 0, 1, 3); // Gradient in Y direction

    // Normalize direction vector (0.7071, 0.7071)
    cv::Vec2f direction(0.7071f, 0.7071f);

    // Apply the embossing effect
    cv::Mat embossed = sobelX * direction[0] + sobelY * direction[1];

    // Normalize the result to 8-bit for display
    double minVal, maxVal;
    cv::minMaxLoc(embossed, &minVal, &maxVal); // Find the range of the result
    embossed.convertTo(dst, CV_8U, 255.0 / (maxVal - minVal), -255.0 * minVal / (maxVal - minVal));

    // Convert single channel to 3 channels
    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR);
}

void pencilSketch(cv::Mat &inputImage, cv::Mat &outputImage) {
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    
    // Invert the grayscale image
    cv::Mat invertedImage;
    cv::bitwise_not(grayImage, invertedImage);
    
    // Blur the output image
    cv::Mat blurredImage;
    cv::GaussianBlur(invertedImage, blurredImage, cv::Size(21, 21), 0);
    
    // Create the pencil sketch by dividing the grayscale image by the blurred output image
    cv::divide(grayImage, 255 - blurredImage, outputImage, 256.0);
    
    // Convert single channel to 3 channels
    cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
}


void oilPainting(cv::Mat &src, cv::Mat &dst) {
    // Apply oil painting effect with radius and color count
    cv::xphoto::oilPainting(src, dst, 10, 1, cv::COLOR_BGR2Lab); 
}