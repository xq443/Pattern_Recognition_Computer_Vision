/*
  Xujia Qin 
  30th Jan, 2025
  S21
*/

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <vector>
#include "csv_util.h"
#include "featureVector.h"
#include "distanceMetric.h"


using namespace std;

/*
 * Given target Image, feature set, feature vectors file computes the
 feature set of target image, reads feature vector file and indentifies top N images.
 */
int main(int argc, char *argv[]) {
    if (argc != 4) { // Expecting 3 arguments: feature set, image path, and number of matches.
        cout << "Usage: ./image_matcher <feature_set> <target_image> <num_matches>" << endl;
        exit(-1);
    }

    string featureset = argv[1]; // Feature set
    char target_image_path[256]; // Store the path of target image.
    char feature_vector_file[256]; // Path of feature vector file.
    int no_of_matches = atoi(argv[3]); // Top N matches to find.

    strcpy(target_image_path, argv[2]); // Assign image path

    vector<char *> filenames; // Vector to store filenames.
    vector<vector<float>> data; // Vectors for data of feature sets.

    // Read image
    cv::Mat targetImage = cv::imread(target_image_path);
    vector<float> targetImageFeatureVector;



    if (featureset == "square") {
        strcpy(feature_vector_file, "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/featureVectors.csv");
        targetImageFeatureVector = sevenXSevenSquare(targetImage);
    } else if (featureset == "hist2D") {
        strcpy(feature_vector_file, "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/hist2DfeatureVector.csv");
        targetImageFeatureVector = twodHistogram(targetImage);
    } else if (featureset == "hist3D") {
        strcpy(feature_vector_file, "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/hist3DfeatureVector.csv");
        targetImageFeatureVector = ThreedHistogram(targetImage);
    } else if (featureset == "multiHist") {
        strcpy(feature_vector_file, "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/multiHistFeaturevectors.csv");
        targetImageFeatureVector = multiHistogram(targetImage);
    } else {
        cerr << "Error: Invalid feature set provided!" << endl;
        return -1;
    }

    // Read feature vector file
    if (read_image_data_csv(feature_vector_file, filenames, data) != 0) {
        cerr << "Error reading image data from CSV file: " << feature_vector_file << endl;
        exit(-1);
    }

    // Compute distances and find top matches
    vector<pair<string, int>> results;
    vector<pair<string, float>> results2;

    if (featureset == "square") {
        results = sum_of_squared_difference(targetImageFeatureVector, data, filenames);
        
        for (int i = 0; i < no_of_matches; i++) {
        cout << results[i].first << ":" << results[i].second << endl;
        cv::imshow(results[i].first, cv::imread(results[i].first));
        }
    } else if (featureset=="hist2D") {
        cout << "Matching hist2d";
        results2 = histogram_intersection(targetImageFeatureVector, data, filenames);
        cout << results2.size();
        for (int i = 0; i < no_of_matches; i++) {
        cout << results2[i].first << ":" << results2[i].second << endl;
        cv::imshow(results2[i].first, cv::imread(results2[i].first));
        }
    } else if (featureset=="hist3D") {
        cout << "Matching hist3d";
        results2 = histogram_intersection(targetImageFeatureVector, data, filenames);
        cout << results2.size();
        for (int i = 0; i < no_of_matches; i++) {
        cout << results2[i].first << ":" << results2[i].second << endl;
        cv::imshow(results2[i].first, cv::imread(results2[i].first));
        }
    } else if (featureset=="multiHist") {
        cout << "Matching top bottom hist";
        results2 = histogram_intersection_for_2histograms(targetImageFeatureVector, data, filenames);
        cout << results2.size();
        for (int i = 0; i < no_of_matches; i++) {
        cout << results2[i].first << ":" << results2[i].second << endl;
        cv::imshow(results2[i].first, cv::imread(results2[i].first));
        }
    }

    // Wait for key press
    if (cv::waitKey(0) == 'q') {
        for (int i = 0; i < no_of_matches; i++) {
        cv::destroyWindow(results[i].first);
        }
    }

    return 0;
}
