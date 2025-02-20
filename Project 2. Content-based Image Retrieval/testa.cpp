#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Compute histogram with depth filtering
void computeHistogram(const Mat& image, const Mat& depthMap, int threshold, bool weightByDepth) {
    vector<int> histogram(256, 0);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int intensity = image.at<uchar>(y, x);
            float depth = depthMap.at<float>(y, x);

            if (depth < threshold) { // Consider only foreground
                if (weightByDepth) {
                    histogram[intensity] += (1.0 / depth); // Weighting by 1/depth
                } else {
                    histogram[intensity]++;
                }
            }
        }
    }

    // Display histogram
    for (int i = 0; i < 256; i++) {
        cout << "Bin " << i << ": " << histogram[i] << endl;
    }
}

int main() {
    // Load grayscale image
    Mat image = imread("olympus/pic.0019.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    // Load depth map (assume it’s a single-channel float image)
    Mat depthMap = imread("DAV2_depthMap.jpg", IMREAD_UNCHANGED);
    if (depthMap.empty()) {
        cerr << "Error loading depth map!" << endl;
        return -1;
    }
    depthMap.convertTo(depthMap, CV_32F); // Ensure float format

    int depthThreshold = 100; // Example threshold
    bool weightByDepth = true;

    computeHistogram(image, depthMap, depthThreshold, weightByDepth);

    return 0;
}
