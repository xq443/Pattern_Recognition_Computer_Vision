/*
  Xujia Qin 30th Jan, 2025
  S21
*/
#include <fstream>
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
bool cmp1(pair<string, float> &a,
		  pair<string, float> &b) {
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

vector<pair<string, float>> histogram_intersection(vector<float> &targetImageFeatures,
												   vector<vector<float>> &featuresData, vector<char *> &filenames) {
  vector<pair<string, float>> results;
  // loop through rows
  for (int i = 0; i < featuresData.size(); i++) {
	float total_difference = 0.0;
	// loop through columsn.
	for (int j = 0; j < featuresData[i].size(); j++) {
	  float difference = min(targetImageFeatures[j], featuresData[i][j]);
	  total_difference += difference;
	}
	results.push_back(make_pair(filenames[i], 1 - total_difference));
  }
  // Sort using comparator function
  sort(results.begin(), results.end(), cmp1);
  return results;

}

vector<pair<string, float>> histogram_intersection_for_2histograms(vector<float> &targetImageFeatures,
																   vector<vector<float>> &featuresData,
																   vector<char *> &filenames) {
  vector<pair<string, float>> results;
  // loop through rows
  for (int i = 0; i < featuresData.size(); i++) {
	float total_difference = 0.0;
	// loop through columsn.
	for (int j = 0; j < featuresData[i].size(); j++) {
	  float difference = min(targetImageFeatures[j], featuresData[i][j]);
	  total_difference += difference;
	}
	results.push_back(make_pair(filenames[i], 2 - total_difference));
  }
  // Sort using comparator function
  sort(results.begin(), results.end(), cmp1);
  return results;
}


vector<pair<string, float>> cosine_distance(vector<float> &targetImageFeatures,
                                             vector<vector<float>> &featuresData,
                                             vector<char *> &filenames) {
    vector<pair<string, float>> results;
    
    if (featuresData.empty() || targetImageFeatures.size() != featuresData[0].size()) {
        cerr << "Error: Feature vector size mismatch!" << endl;
        return results;
    }
    
    for (size_t i = 0; i < featuresData.size(); i++) {
        float dot_product = 0.0, norm_target = 0.0, norm_data = 0.0;
        
        for (size_t j = 0; j < featuresData[i].size(); j++) {
            dot_product += targetImageFeatures[j] * featuresData[i][j];
            norm_target += targetImageFeatures[j] * targetImageFeatures[j];
            norm_data += featuresData[i][j] * featuresData[i][j];
        }
        
        norm_target = sqrt(norm_target);
        norm_data = sqrt(norm_data);
        
        float cosine_similarity = (norm_target > 0 && norm_data > 0) ? dot_product / (norm_target * norm_data) : 0;
        float cosine_distance = 1 - cosine_similarity;
        
        results.emplace_back(filenames[i], cosine_distance);
    }
    
    sort(results.begin(), results.end(), cmp1);
    return results;
}


vector<pair<string, float>> chi_square_distance(vector<float> &targetImageFeatures,
                                             vector<vector<float>> &featuresData,
                                             vector<char *> &filenames) {
    vector<pair<string, float>> results;
    
    if (featuresData.empty() || targetImageFeatures.size() != featuresData[0].size()) {
        cerr << "Error: Feature vector size mismatch!" << endl;
        return results;
    }
    
    for (size_t i = 0; i < featuresData.size(); i++) {
        float chi_square_dist = 0.0;
        
        for (size_t j = 0; j < featuresData[i].size(); j++) {
            float numerator = pow(targetImageFeatures[j] - featuresData[i][j], 2);
            float denominator = targetImageFeatures[j] + featuresData[i][j] + 1e-10; // Avoid division by zero
            chi_square_dist += numerator / denominator;
        }
        
        results.emplace_back(filenames[i], chi_square_dist);
    }
    
    sort(results.begin(), results.end(), cmp1);
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
