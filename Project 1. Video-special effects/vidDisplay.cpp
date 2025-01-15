/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1
 */
#include "filter.h"
#include <opencv2/opencv.hpp>
#include <cstdio> // gives me printf
#include <cstring> // gives me strcpy


int main(int argc, char** argv) {
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

    // Get some properties of the video
    cv::Size frameSize(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    std::cout << "Frame size: " << frameSize.width << "x" << frameSize.height << "\n";

    // Create a window to display the video
    cv::namedWindow("Video Display", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, dstframe;
    bool showGrayscale = false;
    bool showAlternativeGreyscale = false;
    std::cout << "Press 'q' to quit, 's' to save a frame, 'g' to make greyscale, 'h' to make alternative greyscale.\n";

    while (true) {
        cap >> frame; // Capture a new frame

        if (frame.empty()) {
            std::cout << "Captured empty frame.\n";
            break;
        }

        // Check and process the frame to greyscale accordingly
        if (showGrayscale) {
            applyGrayscale(frame); 
        }  
        
        if (showAlternativeGreyscale) {
            if(greyscale(frame, dstframe) == -1) { // if anything goes wrong (-1) by alternative converting way, stop and exit the program
                std::cout << "Alternative grayscale failed.\n";
                break; 
            }
        }
        cv::imshow("Video Display", frame);


        // Check for key presses
        char key = static_cast<char>(cv::waitKey(10)); // Wait for 10 milliseconds

        if (key == 'q') {
            std::cout << "Quit key pressed. Exiting...\n";
            break;
        } else if (key == 's') {
            static int id = 0;
            std::string outputPath = "saved_frame_" + std::to_string(id++) + ".jpg";
            if (cv::imwrite(outputPath, frame)) {
                std::cout << "Saved frame as " << outputPath << "\n";
            } else {
                std::cerr << "Error: Could not save the frame.\n";
            }
        } else if (key == 'g') {
            showGrayscale = !showGrayscale;
            std::cout << "Grayscale mode " << (showGrayscale ? "enabled" : "disabled") << ".\n";
        } else if (key == 'h') {
            showAlternativeGreyscale = !showAlternativeGreyscale;
            std::cout << "Alternative grayscale mode " << (showAlternativeGreyscale ? "enabled" : "disabled") << ".\n";
        }

    }

    // Release resources and close windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}