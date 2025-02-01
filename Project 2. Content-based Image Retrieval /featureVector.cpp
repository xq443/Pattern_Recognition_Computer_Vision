/*
  Xujia Qin 30th Jan, 2025
  S21
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "featureVector.h"
#include <vector>
#include "utils.h"
#include <cmath>

using namespace std;

/*
 * Use the 7x7 square in the middle of the image as a feature vector. 
 * Use sum-of-squared-difference as the distance metric. 
*/


// Function to extract a 7x7 feature vector from the center of an image
vector<float> sevenXSevenSquare(cv::Mat &src) {
    int center_row = src.rows / 2;
    int center_col = src.cols / 2;
    vector<float> featureVector;

    // Ensure image has at least 7x7 pixels
    if (src.rows < 7 || src.cols < 7) {
        cerr << "Error: Image too small for 7x7 extraction!" << endl;
        return featureVector;
    }

    // Row pointers for accessing pixels
    cv::Vec3b *rptrp3 = src.ptr<cv::Vec3b>(center_row + 3);
    cv::Vec3b *rptrp2 = src.ptr<cv::Vec3b>(center_row + 2);
    cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(center_row + 1);
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(center_row);
    cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(center_row - 1);
    cv::Vec3b *rptrm2 = src.ptr<cv::Vec3b>(center_row - 2);
    cv::Vec3b *rptrm3 = src.ptr<cv::Vec3b>(center_row - 3);

    // Loop through the 7x7 window centered at (center_row, center_col)
    for (int col = center_col - 3; col <= center_col + 3; col++) {
        for (int c = 0; c < 2; c++) { // Only using first two color channels
            featureVector.push_back(rptr[col][c]);
            featureVector.push_back(rptrp1[col][c]);
            featureVector.push_back(rptrp2[col][c]);
            featureVector.push_back(rptrp3[col][c]);
            featureVector.push_back(rptrm1[col][c]);
            featureVector.push_back(rptrm2[col][c]);
            featureVector.push_back(rptrm3[col][c]);
        }
    }

    return featureVector;
}

// int main() {
//     // Load an image (modify the path as needed)
//     cv::Mat img = cv::imread("olympus/pic.0666.jpg");

//     // Check if the image loaded successfully
//     if (img.empty()) {
//         cerr << "Error: Could not load the image!" << endl;
//         return -1;
//     }

//     // Extract the 7x7 feature vector
//     vector<float> featureVector = sevenXSevenSquare(img);

//     // Check if the feature vector was extracted successfully
//     if (featureVector.empty()) {
//         cerr << "Error: Feature extraction failed!" << endl;
//         return -1;
//     }

//     // Print feature vector values
//     cout << "Feature Vector (size: " << featureVector.size() << "):" << endl;
//     for (size_t i = 0; i < featureVector.size(); i++) {
//         cout << featureVector[i] << " ";
//         if ((i + 1) % 7 == 0) cout << endl; // Format output every 7 elements
//     }

//     return 0;
// }
