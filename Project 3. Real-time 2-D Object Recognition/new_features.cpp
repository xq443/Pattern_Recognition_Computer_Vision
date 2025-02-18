#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// Function to compute the Euclidean distance with scaled features
double computeEuclideanDistance(const vector<double>& feature1, const vector<double>& feature2, const vector<double>& stdev) {
    double distance = 0.0;
    for (size_t i = 0; i < feature1.size(); i++) {
        // Scale the feature by its standard deviation
        double diff = (feature1[i] - feature2[i]) / stdev[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

// Function to load the object database from a CSV file
void loadObjectDatabase(const string& filename, vector<vector<double>>& featureVectors, vector<string>& labels) {
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string label;
        getline(ss, label, ',');
        labels.push_back(label);
        
        vector<double> features;
        string feature;
        while (getline(ss, feature, ',')) {
            features.push_back(stod(feature));
        }
        featureVectors.push_back(features);
    }
}

// Function to calculate the standard deviation for each feature in the database
vector<double> calculateStandardDeviations(const vector<vector<double>>& featureVectors) {
    size_t numFeatures = featureVectors[0].size();
    vector<double> stdev(numFeatures, 0.0);
    size_t numVectors = featureVectors.size();
    
    // Calculate mean for each feature
    vector<double> means(numFeatures, 0.0);
    for (const auto& features : featureVectors) {
        for (size_t i = 0; i < numFeatures; i++) {
            means[i] += features[i];
        }
    }
    for (size_t i = 0; i < numFeatures; i++) {
        means[i] /= numVectors;
    }

    // Calculate standard deviation
    for (const auto& features : featureVectors) {
        for (size_t i = 0; i < numFeatures; i++) {
            stdev[i] += pow(features[i] - means[i], 2);
        }
    }
    for (size_t i = 0; i < numFeatures; i++) {
        stdev[i] = sqrt(stdev[i] / numVectors);
    }
    
    return stdev;
}

// Function to classify an unknown object by finding the closest feature vector in the object DB
string classifyObject(const vector<double>& unknownFeatureVector, const vector<vector<double>>& featureVectors, const vector<string>& labels, const vector<double>& stdev) {
    double minDistance = numeric_limits<double>::infinity();
    string closestLabel;
    
    for (size_t i = 0; i < featureVectors.size(); i++) {
        double dist = computeEuclideanDistance(unknownFeatureVector, featureVectors[i], stdev);
        if (dist < minDistance) {
            minDistance = dist;
            closestLabel = labels[i];
        }
    }
    return closestLabel;
}

// Function to handle real-time classification and display the label on the output
void classifyAndDisplay(Mat& frame, const Mat& binary, const vector<vector<double>>& featureVectors, const vector<string>& labels, const vector<double>& stdev) {
    Mat labelsMat, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labelsMat, stats, centroids);

    // Display the regions in the frame
    for (int i = 1; i < numLabels; i++) { // Skip background
        vector<double> featureVector = computeRegionFeatures(binary, i);
        
        // Classify the unknown object
        string label = classifyObject(featureVector, featureVectors, labels, stdev);

        // Visualize the region on the frame with its label
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        rectangle(frame, Point(x, y), Point(x + width, y + height), Scalar(255, 0, 0), 2);
        putText(frame, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
    }

    // Display the updated frame with the classification
    imshow("Classified Video", frame);
}

int main() {
    // Load the object database from file
    vector<vector<double>> featureVectors;
    vector<string> labels;
    loadObjectDatabase("object_db.csv", featureVectors, labels);

    // Calculate the standard deviations for feature normalization
    vector<double> stdev = calculateStandardDeviations(featureVectors);

    VideoCapture cap(0); // Open webcam
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera!" << endl;
        return -1;
    }

    namedWindow("Original Video", WINDOW_AUTOSIZE / 2);
    namedWindow("Thresholded Video", WINDOW_AUTOSIZE / 2);

    while (true) {
        Mat frame;
        cap >> frame; // Capture frame
        if (frame.empty()) break;

        Mat gray = convertToGray(frame);
        Mat blurred = applyBlur(gray);
        int threshold = computeDynamicThreshold(blurred);
        Mat binary = thresholdImage(blurred, threshold);

        // Display thresholded video
        imshow("Thresholded Video", binary);

        // Handle classification of unknown objects
        classifyAndDisplay(frame, binary, featureVectors, labels, stdev);

        // Press ESC to exit
        char key = waitKey(1);
        if (key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
