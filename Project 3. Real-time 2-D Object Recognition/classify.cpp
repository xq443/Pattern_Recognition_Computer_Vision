#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

struct FeatureVector {
    std::string label;
    std::vector<float> features;
};

// Function to compute standard deviation for each feature column
std::vector<float> computeStdDev(const std::vector<FeatureVector>& dataset) {
    int featureSize = dataset[0].features.size();
    std::vector<float> mean(featureSize, 0.0f);
    std::vector<float> stddev(featureSize, 0.0f);
    int n = dataset.size();

    // Compute mean
    for (const auto& data : dataset) {
        for (size_t i = 0; i < featureSize; ++i) {
            mean[i] += data.features[i];
        }
    }
    for (auto& val : mean) val /= n;

    // Compute standard deviation
    for (const auto& data : dataset) {
        for (size_t i = 0; i < featureSize; ++i) {
            stddev[i] += std::pow(data.features[i] - mean[i], 2);
        }
    }
    for (auto& val : stddev) val = std::sqrt(val / (n - 1));
    
    return stddev;
}

// Function to compute scaled Euclidean distance
float computeDistance(const std::vector<float>& f1, const std::vector<float>& f2, const std::vector<float>& stddev) {
    float distance = 0.0f;
    for (size_t i = 0; i < f1.size(); ++i) {
        float diff = (f1[i] - f2[i]) / stddev[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

// Function to extract features from a new image
std::vector<float> extractFeatures(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cerr << "Error: No object detected in image!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cv::Moments m = cv::moments(contours[0]);
    float cx = m.m10 / m.m00;
    float cy = m.m01 / m.m00;
    
    cv::Rect boundingBox = cv::boundingRect(contours[0]);
    float boundingRatio = (float)boundingBox.width / boundingBox.height;
    float percentFilled = (float)cv::contourArea(contours[0]) / (boundingBox.width * boundingBox.height);
    float theta = 0.5 * std::atan2(2 * m.mu11, m.mu20 - m.mu02) * (180.0 / CV_PI);

    return {cx, cy, boundingRatio, percentFilled, theta};
}

// Function to classify new image based on database
std::string classifyImage(const std::string& imagePath, const std::string& dbPath) {
    std::ifstream file(dbPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open database file!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::vector<FeatureVector> dataset;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label;
        std::getline(ss, label, ',');
        
        std::vector<float> features;
        std::string value;
        while (std::getline(ss, value, ',')) {
            features.push_back(std::stof(value));
        }
        dataset.push_back({label, features});
    }
    file.close();
    
    std::vector<float> stddev = computeStdDev(dataset);
    std::vector<float> newImageFeatures = extractFeatures(imagePath);
    
    float minDistance = std::numeric_limits<float>::max();
    std::string bestMatch;
    
    for (const auto& data : dataset) {
        float distance = computeDistance(newImageFeatures, data.features, stddev);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = data.label;
        }
    }
    
    return bestMatch;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <db_path>" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::string imagePath = argv[1];
    std::string dbPath = argv[2];
    
    std::string resultLabel = classifyImage(imagePath, dbPath);
    std::cout << "Predicted Label: " << resultLabel << std::endl;
    
    // Load the image and display it with the label
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return EXIT_FAILURE;
    }
    
    // Put the label on the image
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 2.0;
    int thickness = 3;
    cv::Point textOrg(50, 150);
    cv::putText(img, resultLabel, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
    
    // Show the image
    cv::imshow("Classified Image", img);
    cv::waitKey(0);
    
    return EXIT_SUCCESS;
}
