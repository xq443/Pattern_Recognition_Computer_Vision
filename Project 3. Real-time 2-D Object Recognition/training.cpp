/**
 * Xujia Qin 20th Feb, 2025
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>

using namespace cv;
using namespace std;

// Function to convert BGR to grayscale manually
Mat convertToGray(const Mat &frame) {
    Mat gray(frame.rows, frame.cols, CV_8UC1);
    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            Vec3b pixel = frame.at<Vec3b>(i, j);
            uint8_t grayValue = static_cast<uint8_t>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            gray.at<uint8_t>(i, j) = grayValue;
        }
    }
    return gray;
}

// Function to apply a simple box blur
Mat applyBlur(const Mat &image) {
    Mat blurred(image.rows, image.cols, CV_8UC1);
    int kernel_size = 3;
    int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int sum_kernel = 16;

    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            int sum = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += image.at<uint8_t>(i + ki, j + kj) * kernel[ki + 1][kj + 1];
                }
            }
            blurred.at<uint8_t>(i, j) = sum / sum_kernel;
        }
    }
    return blurred;
}

// Function to compute dynamic threshold using K-means (K=2)
int computeDynamicThreshold(const Mat &gray) {
    vector<int> pixels;
    int sampleRate = 16; // Use 1/16 of pixels for faster computation
    RNG rng; // random generator

    for (int i = 0; i < gray.rows; i += sampleRate) {
        for (int j = 0; j < gray.cols; j += sampleRate) {
            pixels.push_back(gray.at<uint8_t>(i, j));
        }
    }

    // K-means clustering with K=2
    int mean1 = pixels[rng.uniform(0, (int)pixels.size())];
    int mean2 = pixels[rng.uniform(0, (int)pixels.size())];

    for (int iter = 0; iter < 5; iter++) { // Iterate a few times for convergence
        vector<int> cluster1, cluster2;
        for (int val : pixels) {
            if (abs(val - mean1) < abs(val - mean2))
                cluster1.push_back(val);
            else
                cluster2.push_back(val);
        }
        if (!cluster1.empty()) mean1 = accumulate(cluster1.begin(), cluster1.end(), 0) / cluster1.size();
        if (!cluster2.empty()) mean2 = accumulate(cluster2.begin(), cluster2.end(), 0) / cluster2.size();
    }

    return (mean1 + mean2) / 2; // Return midpoint as the threshold
}

// Function to threshold an image using a given threshold
Mat thresholdImage(const Mat &gray, int threshold) {
    Mat binary(gray.rows, gray.cols, CV_8UC1);
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            binary.at<uint8_t>(i, j) = (gray.at<uint8_t>(i, j) > threshold) ? 255 : 0;
        }
    }
    return binary;
}

// Function to compute region features (bounding box, centroid, moments)
vector<double> computeRegionFeatures(const Mat &regionMap, int regionID) {
    // Find connected components and stats
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(regionMap, labels, stats, centroids);

    // Extract stats for the given region
    int x = stats.at<int>(regionID, CC_STAT_LEFT);
    int y = stats.at<int>(regionID, CC_STAT_TOP);
    int width = stats.at<int>(regionID, CC_STAT_WIDTH);
    int height = stats.at<int>(regionID, CC_STAT_HEIGHT);
    int area = stats.at<int>(regionID, CC_STAT_AREA);

    // Extract centroid
    double cx = centroids.at<double>(regionID, 0);
    double cy = centroids.at<double>(regionID, 1);

    // Create a mask for the region
    Mat regionMask = (labels == regionID);

    // Compute moments
    Moments moment = moments(regionMask, false);
    
    // Calculate the central moments (mu20, mu02, mu11)
    double mu20 = moment.mu20;
    double mu02 = moment.mu02;
    double mu11 = moment.mu11;
    
    // Calculate the angle of the axis of least central moment
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
    
    // Bounding box aspect ratio (width / height)
    double boundingBoxRatio = static_cast<double>(width) / height;

    // Percent filled (area vs bounding box area)
    double boundingBoxArea = static_cast<double>(width) * height;
    double percentFilled = static_cast<double>(area) / boundingBoxArea;

    // Create the feature vector
    vector<double> featureVector = {cx, cy, boundingBoxRatio, percentFilled, theta * 180 / CV_PI}; // Angle in degrees
    return featureVector;
}

// Function to store the feature vector and label in a file
void storeFeatureVector(const vector<double> &featureVector, const string &label, const string &filename = "object_db.csv") {
    ofstream outFile(filename, ios::app); // Open in append mode
    if (outFile.is_open()) {
        outFile << label;
        for (double feature : featureVector) {
            outFile << "," << feature;
        }
        outFile << endl;
        outFile.close();
    } else {
        cerr << "Error opening file for writing!" << endl;
    }
}

// Function to handle the training mode (labeling objects)
void trainObject(Mat &frame, const Mat &binary) {
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labels, stats, centroids);

    // Display the regions in the frame
    for (int i = 1; i < numLabels; i++) { // Skip background
        vector<double> featureVector = computeRegionFeatures(binary, i);

        // Display the region and ask for user input
        cout << "Object detected, please enter a label for it: ";
        string label;
        cin >> label;

        // Store the feature vector and label
        storeFeatureVector(featureVector, label);

        // Visualize the region on the frame
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        rectangle(frame, Point(x, y), Point(x + width, y + height), Scalar(255, 0, 0), 2);
        putText(frame, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
    }

    // Display the updated frame
    imshow("Training Mode", frame);
}

int main() {
    // Load the PNG image
    Mat frame = imread("/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 3. Real-time 2-D Object Recognition/db/f5.png");  // Change the file path accordingly
    if (frame.empty()) {
        cerr << "Error: Cannot open image!" << endl;
        return -1;
    }

    // Process the image as before
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Thresholded Image", WINDOW_AUTOSIZE);

    Mat gray = convertToGray(frame);
    Mat blurred = applyBlur(gray);
    int threshold = computeDynamicThreshold(blurred);
    Mat binary = thresholdImage(blurred, threshold);

    // Display original and thresholded image
    imshow("Original Image", frame);
    imshow("Thresholded Image", binary);

    // Wait for user input to enter training mode
    char key = waitKey(0);  // Wait indefinitely for key press
    if (key == 'n' || key == 'N') {
        trainObject(frame, binary);
    }

    // Press ESC to exit
    if (key == 27) {
        destroyAllWindows();
        return 0;
    }

    destroyAllWindows();
    return 0;
}