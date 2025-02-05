#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "featureVector.h"
#include "utils.h"
#include <cmath>
#include <vector>
#include <string>
#include <cstring> // For strcmp
using namespace std;

// Function to compute a multi 3D-histogram for a given image, using depth map to filter pixels
vector<float> depthFilteredMultiHistogram(cv::Mat &src, cv::Mat &depthMap, int bins, float depthThreshold) {
    vector<vector<vector<int>>> hist3d1(
        bins + 1, vector<vector<int>>(bins + 1, vector<int>(bins + 1, 0)));
    vector<float> result;
    int bin_size = 255 / bins;
    float total_pixels = 0.0;

    // Top half
    for (int i = 0; i <= src.rows / 2; i++) {
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        float *dptr = depthMap.ptr<float>(i);
        for (int j = 0; j < src.cols; j++) {
            if (dptr[j] > depthThreshold) continue; // Skip pixels beyond the depth threshold

            float blue = rptr[j][0];
            float green = rptr[j][1];
            float red = rptr[j][2];

            int blue_index = blue / bin_size;
            int green_index = green / bin_size;
            int red_index = red / bin_size;

            total_pixels++;
            hist3d1[blue_index][green_index][red_index] += 1;
        }
    }

    // Bottom half
    vector<vector<vector<int>>> hist3d2(
        bins + 1, vector<vector<int>>(bins + 1, vector<int>(bins + 1, 0)));

    for (int i = (src.rows / 2) + 1; i < src.rows; i++) {
        cv::Vec3b *rptr1 = src.ptr<cv::Vec3b>(i);
        float *dptr1 = depthMap.ptr<float>(i);
        for (int j = 0; j < src.cols; j++) {
            if (dptr1[j] > depthThreshold) continue; // Skip pixels beyond the depth threshold

            float blue = rptr1[j][0];
            float green = rptr1[j][1];
            float red = rptr1[j][2];

            int blue_index = blue / bin_size;
            int green_index = green / bin_size;
            int red_index = red / bin_size;

            hist3d2[blue_index][green_index][red_index] += 1;
            total_pixels++;
        }
    }

    // Normalize and flatten the histograms
    for (int i = 0; i < bins + 1; i++) {
        for (int j = 0; j < bins + 1; j++) {
            for (int k = 0; k < bins + 1; k++) {
                result.push_back(hist3d1[i][j][k] / total_pixels);
            }
        }
    }

    for (int i = 0; i < bins + 1; i++) {
        for (int j = 0; j < bins + 1; j++) {
            for (int k = 0; k < bins + 1; k++) {
                result.push_back(hist3d2[i][j][k] / total_pixels);
            }
        }
    }

    return result;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <RGB image (JPG)> <Depth map (PNG)>" << std::endl;
        return -1;
    }

    // Load the RGB image (JPG format)
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Unable to read RGB image " << argv[1] << std::endl;
        return -1;
    }

    // Load the depth map (PNG format)
    cv::Mat depthMap = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    if (depthMap.empty()) {
        std::cerr << "Error: Unable to read depth map " << argv[2] << std::endl;
        return -1;
    }

    // Convert multi-channel depth maps to single-channel grayscale
    if (depthMap.channels() == 3) {
        cv::cvtColor(depthMap, depthMap, cv::COLOR_BGR2GRAY);
    } else if (depthMap.channels() == 4) {
        cv::cvtColor(depthMap, depthMap, cv::COLOR_BGRA2GRAY);
    }

    // Convert depth map to floating point (normalize if necessary)
    if (depthMap.type() == CV_16U) {
        depthMap.convertTo(depthMap, CV_32F, 1.0 / 65535.0); // Normalize 16-bit to [0,1]
    } else if (depthMap.type() == CV_8U) {
        depthMap.convertTo(depthMap, CV_32F, 1.0 / 255.0); // Normalize 8-bit to [0,1]
    } else if (depthMap.type() != CV_32F) {
        std::cerr << "Unsupported depth map format: " << depthMap.type() << std::endl;
        return -1;
    }

    // Ensure depth map and image have the same dimensions
    if (src.size() != depthMap.size()) {
        std::cerr << "Error: RGB image and depth map dimensions do not match!" << std::endl;
        return -1;
    }

    // Set parameters
    int bins = 8;
    float depthThreshold = 0.51; // Adjust as needed

    // Compute the depth-filtered histogram
    std::vector<float> histogram = depthFilteredMultiHistogram(src, depthMap, bins, depthThreshold);

    // Print first 10 histogram values for debugging
    std::cout << "Histogram values (first 10): ";
    for (size_t i = 0; i < histogram.size(); ++i) {
        std::cout << histogram[i] << " ";
    }
    std::cout << "..." << std::endl;

    return 0;
}