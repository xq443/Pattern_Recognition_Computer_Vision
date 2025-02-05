#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
#include "featureVector.h"
#include "utils.h"
#include <cmath>
#include <vector>
#include <string>
#include <cstring> // For strcmp

using namespace std;

// Function to compute a depth map using the DA2Network
cv::Mat computeDepthMap(cv::Mat &src) {
    cv::Mat dst;
    DA2Network da_net("model_fp16.onnx");

    float scale_factor = 512.0 / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

    da_net.set_input(src, scale_factor);
    da_net.run_network(dst, src.size());

    dst = dst * 5.0; // Scale depth values by 5
    return dst;
}

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
    if (argc < 2) {
        printf("Usage %s <image filename>\n", argv[0]);
        exit(-1);
    }

    char filename[256];
    strcpy(filename, argv[1]);

    cv::Mat src = cv::imread(filename);
    if (src.data == NULL) {
        printf("Unable to read image %s\n", filename);
        exit(-1);
    }

    cv::Mat depthMap = computeDepthMap(src);
    float depthThreshold = 50.0;

    vector<float> featureVector = depthFilteredMultiHistogram(src, depthMap, 8, depthThreshold);

    cout << "Feature Vector (size: " << featureVector.size() << "):" << endl;
    for (size_t i = 0; i < featureVector.size(); i++) {
        cout << featureVector[i] << " ";
        if ((i + 1) % 8 == 0) cout << endl;
    }

    return 0;
}