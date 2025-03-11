#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Check if an image path is provided as a command-line argument
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image from the provided path
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cout << "Could not load the image: " << argv[1] << std::endl;
        return -1;
    }

    // Set up ArUco detector parameters
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    detectorParams.adaptiveThreshWinSizeMin = 3;
    detectorParams.adaptiveThreshWinSizeMax = 23;
    detectorParams.adaptiveThreshConstant = 7;

    // Use the 4x4 dictionary with 50 markers
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    bool showRejected = true; // Set to false if you don't want to see rejected markers

    // Convert the image to grayscale (optional, but can improve detection)
    cv::Mat gray_frame;
    cv::cvtColor(image, gray_frame, cv::COLOR_BGR2GRAY);

    // Detect ArUco markers in the image
    vector<int> ids;
    vector<vector<Point2f>> corners, rejected;

    detector.detectMarkers(image, corners, ids, rejected);

    // Print the number of markers detected and rejected
    std::cout << "Number of markers detected: " << ids.size() << std::endl;
    std::cout << "Number of rejected candidates: " << rejected.size() << std::endl;

    // Create a copy of the image to draw the results
    cv::Mat imageCopy;
    image.copyTo(imageCopy);

    // If markers are detected, draw them and their IDs
    if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

        // Draw the ID of each marker
        for (size_t i = 0; i < ids.size(); i++) {
            cv::Point2f topLeft = corners[i][0];
            std::string idText = std::to_string(ids[i]);
            cv::putText(imageCopy, idText, topLeft, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }

    // If enabled, draw rejected candidates (useful for debugging)
    if (showRejected && !rejected.empty()) {
        cv::aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));
    }

    // Display the output image
    imshow("Aruco Marker Detection", imageCopy);

    // Save the output image if 's' is pressed
    char key = (char)waitKey(0);
    if (key == 's') {
        imwrite("output.png", imageCopy);
        cout << "Image saved as output.png" << endl;
    }

    return 0;
}