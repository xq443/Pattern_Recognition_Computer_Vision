#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <string>
#include <cstring> // For strcmp
#include <algorithm> // For std::min

using namespace std;

// Function to compute a multi 3D-histogram for a given image, using depth map to filter pixels
vector<float> depthFilteredMultiHistogram(cv::Mat &src, cv::Mat &depthMap, int bins, float depthThreshold) {
    vector<vector<vector<int>>> hist3d1(
        bins + 1, vector<vector<int>>(bins + 1, vector<int>(bins + 1, 0)));
    vector<float> result;
    int bin_size = 255 / bins;
    float total_pixels = 0.0;

    // Ensure depthMap is valid and has the same size as src
    if (src.size() != depthMap.size()) {
        cout << "Error: Image size and depth map size do not match!" << endl;
        exit(-1);
    }

    // Top half
    for (int i = 0; i <= src.rows / 2; i++) {
        const cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);  // rptr as const
        const float *dptr = depthMap.ptr<float>(i);     // dptr as const
        for (int j = 0; j < src.cols; j++) {
            if (dptr[j] > depthThreshold) continue; // Skip pixels beyond the depth threshold

            float blue = rptr[j][0];
            float green = rptr[j][1];
            float red = rptr[j][2];

            int blue_index = blue / (255 / bins);
            int green_index = green / (255 / bins);
            int red_index = red / (255 / bins);

            // Ensure the indices are within valid bounds
            blue_index = std::min(blue_index, bins);
            green_index = std::min(green_index, bins);
            red_index = std::min(red_index, bins);

            total_pixels++;
            hist3d1[blue_index][green_index][red_index] += 1;
        }
    }

    // Bottom half
    vector<vector<vector<int>>> hist3d2(
        bins + 1, vector<vector<int>>(bins + 1, vector<int>(bins + 1, 0)));

    for (int i = (src.rows / 2) + 1; i < src.rows; i++) {
        const cv::Vec3b *rptr1 = src.ptr<cv::Vec3b>(i);  // rptr1 as const
        const float *dptr1 = depthMap.ptr<float>(i);     // dptr1 as const
        for (int j = 0; j < src.cols; j++) {
            if (dptr1[j] > depthThreshold) continue; // Skip pixels beyond the depth threshold

            float blue = rptr1[j][0];
            float green = rptr1[j][1];
            float red = rptr1[j][2];

            int blue_index = blue / (255 / bins);
            int green_index = green / (255 / bins);
            int red_index = red / (255 / bins);

            // Ensure the indices are within valid bounds
            blue_index = std::min(blue_index, bins);
            green_index = std::min(green_index, bins);
            red_index = std::min(red_index, bins);

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
        printf("Usage %s <image filename> <depth map filename>\n", argv[0]);
        exit(-1);
    }

    char filename[256], depthMapFilename[256];
    strcpy(filename, argv[1]);
    strcpy(depthMapFilename, argv[2]);

    // Load the image
    cv::Mat src = cv::imread(filename);
    if (src.empty()) {
        printf("Unable to read image %s\n", filename);
        exit(-1);
    }

    // Load the depth map
    cv::Mat depthMap = cv::imread(depthMapFilename, cv::IMREAD_UNCHANGED);
    if (depthMap.empty()) {
        printf("Unable to read depth map %s\n", depthMapFilename);
        exit(-1);
    }

    // Debug output: Check image size
    cout << "Image loaded: " << src.rows << "x" << src.cols << endl;
    cout << "Depth map loaded: " << depthMap.rows << "x" << depthMap.cols << endl;

    // Ensure depth map is single channel (grayscale) and convert if necessary
    if (depthMap.channels() > 1) {
        cv::cvtColor(depthMap, depthMap, cv::COLOR_BGR2GRAY);
    }

    // Normalize the depth map (if necessary, depending on the depth map format)
    depthMap.convertTo(depthMap, CV_32F, 1.0 / 255.0);  // Scale to [0, 1] if needed

    // Compute the feature vector using the depth map
    float depthThreshold = 0.05;
    vector<float> featureVector = depthFilteredMultiHistogram(src, depthMap, 8, depthThreshold);

    // Compute the weighted average
    //float weighted_average = weightedAvg(featureVector);

    // Save to CSV file
    // ofstream outFile("output.csv", ios::app);  // Open file in append mode
    // if (outFile.is_open()) {
    //     outFile << filename << "," << weighted_average << endl;
    //     outFile.close();
    // } else {
    //     cout << "Error opening CSV file for writing!" << endl;
    // }

   // cout << "Weighted Average: " << weighted_average << endl;

    return 0;
}
