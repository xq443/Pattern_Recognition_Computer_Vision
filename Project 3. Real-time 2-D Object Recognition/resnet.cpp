/**
 * Xujia Qin 20th Feb, 2025
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

int main() {
    // Load the ONNX model
    std::string model = "resnet50_v1.onnx"; 
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model);
    
    if (net.empty()) {
        std::cerr << "Error: Could not load the network!" << std::endl;
        return -1;
    }

    // Load the class labels
    std::string labelsFile = "labels.txt";
    std::vector<std::string> classLabels;
    std::ifstream ifs(labelsFile.c_str());
    std::string line;
    while (std::getline(ifs, line)) {
        classLabels.push_back(line);
    }

    // Load the image
    std::string imagePath = "db/newf1.png"; 
    cv::Mat image = cv::imread(imagePath);
    
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    // Convert the image to RGB if it's not already
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    // Prepare the image for the network
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224, 224), cv::Scalar(104, 117, 123), false, false);
    net.setInput(blob);

    // Run forward pass to get the output
    cv::Mat prob = net.forward();
    if (prob.empty()) {
        std::cerr << "Error: Forward pass failed!" << std::endl;
        return -1;
    }

    // Get the class with the highest score
    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(prob.reshape(1, 1), 0, &maxVal, 0, &maxLoc);
    int classId = maxLoc.x;

    // Print the predicted class label
    std::cout << "Predicted Label: " << classLabels[classId] << std::endl;

    // Display the image with the predicted label
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    cv::putText(image, classLabels[classId], cv::Point(30, 30), fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);

    // Show the image
    cv::imshow("Image Classification", image);
    cv::waitKey(0);

    return 0;
}
