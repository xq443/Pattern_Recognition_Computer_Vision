/*
  Xujia Qin 30th Jan, 2025
  S21
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "distanceMetric.h"
#include "featureVector.h"

// Comparator function
bool cmp(pair<string, int> &a, pair<string, int> &b) {
    return a.second < b.second;
}

// Function to compute sum-of-squared differences
vector<pair<string, int>> sum_of_squared_difference(vector<float> &targetImageFeatures,
                                                    vector<vector<float>> &featuresData,
                                                    vector<char *> &filenames) {
    vector<pair<string, int>> results;

    // Ensure input dimensions match
    if (featuresData.empty() || targetImageFeatures.size() != featuresData[0].size()) {
        cerr << "Error: Feature vector size mismatch!" << endl;
        return results;
    }

    // Loop through all images in the database
    for (size_t i = 0; i < featuresData.size(); i++) {
        int total_difference = 0;
        
        // Compute sum of squared differences
        for (size_t j = 0; j < featuresData[i].size(); j++) {
            int difference = targetImageFeatures[j] - featuresData[i][j];
            total_difference += difference * difference;
        }

        results.emplace_back(filenames[i], total_difference);
    }

    // Sort results by ascending distance
    sort(results.begin(), results.end(), cmp);

    return results;
}

// functional test
// int main() {
//     // List of image filenames
//     vector<string> filenames = {
//         "olympus/pic.0442.jpg", 
//         "olympus/pic.0443.jpg", 
//         "olympus/pic.0444.jpg"
//     };

//     // Load target image and extract feature vector
//     cv::Mat targetImage = cv::imread("olympus/pic.0441.jpg");  // Change to your actual target image
//     if (targetImage.empty()) {
//         cerr << "Error: Could not load target image!" << endl;
//         return -1;
//     }
//     vector<float> targetImageFeatures = sevenXSevenSquare(targetImage);

//     // Ensure target image feature extraction was successful
//     if (targetImageFeatures.empty()) {
//         cerr << "Error: Could not extract features from target image!" << endl;
//         return -1;
//     }

//     // Extract feature vectors for dataset images
//     vector<vector<float>> featuresData;
//     for (const string &filename : filenames) {
//         cv::Mat img = cv::imread(filename);
//         if (img.empty()) {
//             cerr << "Error: Could not load image " << filename << "!" << endl;
//             continue;
//         }
//         vector<float> features = sevenXSevenSquare(img);
//         if (!features.empty()) {
//             featuresData.push_back(features);
//         }
//     }

//     // Ensure we have extracted valid feature vectors
//     if (featuresData.empty()) {
//         cerr << "Error: No valid feature vectors extracted!" << endl;
//         return -1;
//     }

//     // Compute sum-of-squared differences
//     vector<pair<string, int>> results = sum_of_squared_difference(targetImageFeatures, featuresData, filenames);

//     // Print sorted results
//     cout << "Sorted Image Distances:" << endl;
//     for (const auto &result : results) {
//         cout << "Filename: " << result.first << ", Distance: " << result.second << endl;
//     }

//     return 0;
// }
