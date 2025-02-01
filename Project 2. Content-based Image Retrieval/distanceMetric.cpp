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

/*
 * Function that computes the histogram_intersection distance for all the images in the database,
 * and returns the distances as pair of vector with (filename, distance) in sorted order.

      Arg1: targetImageFeature: A vector that has the features of target Image.
      Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
      Arg3: filenames: A vector containing paths of all the image in database.

      returns a sorted vector with (filename, distance) pairs.
 */
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

/*
 * Function that computes the histogram_intersection distance(for multiple histogram features) for all the images in the database,
 * and returns the distances as pair of vector with (filename, distance) in sorted order.

      Arg1: targetImageFeature: A vector that has the features of target Image.
      Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
      Arg3: filenames: A vector containing paths of all the image in database.

      returns a sorted vector with (filename, distance) pairs.
 */
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

/*
 * Funtions that computes the entropy distance for all the images in the database with target Image.
 * and returns the distances as a pair of vector with (filename, distance) in sorted order.

        Arg1: targetImageFeature: A vector that has the features of target Image.
        Arg2: featuresData: A 2 dimensional vector containing features of all Images in dataset.
        Arg3: filenames: A vector containing paths of all the image in database.

        returns a sorted vector with (filename, distance) pairs.
 */
vector<pair<string, float>> entopyDistance(vector<float> &targetImageFeatures,
										   vector<vector<float>> &featuresData,
										   vector<char *> &filenames) {
  vector<pair<string, float>> results;
  for (int i = 0; i < featuresData.size(); i++) {
	float total_entropy1 = 0.0;
	float total_entropy2 = 0.0;
	// loop through columsn.
	for (int j = 0; j < featuresData[i].size(); j++) {
	  float curr_probability1 = 0.0;
	  float curr_probablity2 = 0.0; // computing pi * log(pi) for target and matching image.
	  if (targetImageFeatures[j]!=0) {
		curr_probability1 = (targetImageFeatures[j])*(::log(targetImageFeatures[j]));
	  }
	  if (featuresData[i][j]!=0) {
		curr_probablity2 = (featuresData[i][j])*(::log(featuresData[i][j]));
	  }

	  //cout << curr_probability1 << "," << curr_probablity2;
	  // incrememt total entropy by current probability.
	  total_entropy1 += curr_probability1;
	  total_entropy2 += curr_probablity2;
	}
	float entropy_difference = (-1*total_entropy1) - (-1*total_entropy2);
	//cout << entropy_difference << endl;
	results.push_back(make_pair(filenames[i], entropy_difference));
  }
  // Sort using comparator function
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
