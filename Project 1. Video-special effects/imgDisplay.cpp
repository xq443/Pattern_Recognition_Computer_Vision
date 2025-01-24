/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1
 */

#include <opencv2/opencv.hpp>
#include <cstdio> 

int main(int argc, char *argv[]) {
    cv::VideoCapture cap; 

    // Check if a video file is provided as a command-line argument
    if (argc > 1) {
        std::string videoPath = argv[1];
        cap.open(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video file: " << videoPath << "\n";
            return 1;
        }
    } else {
        // Default to webcam if no argument is provided
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video capture device.\n";
            return 1;
        }
    }

    cv::Mat frame;
    std::cout << "Press 'q' to quit.\n";

    while (true) {
        cap >> frame; // Capture a new frame from the camera
        if (frame.empty()) {
            printf("Error: Unable to capture frame\n");
            break;
        }

        cv::imshow("Video Display", frame); // Display the frame

        int key = cv::waitKey(1); // Check for key presses with minimal delay
        if (key == 'q') {
            std::cout << "Quit key pressed. Exiting...\n";
            break;
        } else if (key == 's') {
            std::string filename = "saved_image.jpg";
            cv::imwrite(filename, frame); // Save the current frame to a file
            std::cout << "Image saved as " << filename << "\n";
        }
    }

    cap.release(); // Release the camera
    cv::destroyAllWindows();
    printf("Terminating\n");
    return 0;
}