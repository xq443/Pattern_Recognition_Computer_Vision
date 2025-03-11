#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>

int main() {
    // Open the camera
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Cannot open the video camera" << std::endl;
        return -1;
    }

    // ARUCO Marker dictionary
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();

    // Create the ArucoDetector object
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    // Loop to continuously capture frames
    while (true) {
        cv::Mat frame;
        bool bsuccess = cap.read(frame);

        if (!bsuccess) {
            std::cout << "Video camera is disconnected" << std::endl;
            break;
        }

        // Detect markers in the frame
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        detector.detectMarkers(frame, markerCorners, markerIds);

        // Draw detected markers
        cv::Mat imagecopy;
        frame.copyTo(imagecopy);
        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(imagecopy, markerCorners, markerIds);

            // Draw the marker ID at the top-left corner of each detected marker
            for (size_t i = 0; i < markerIds.size(); i++) {
                // Get the top-left corner of the marker
                cv::Point2f topLeft = markerCorners[i][0];

                // Convert marker ID to string and draw it on the image
                std::string idText = std::to_string(markerIds[i]);
                cv::putText(imagecopy, idText, topLeft, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
        }

        // Display the result
        cv::imshow("Detected Aruco Markers", imagecopy);

        // Exit the loop when the user presses 'q'
        int key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    return 0;
}
