/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1
 */
#include "filter.h"
#include "timeBlur.h" 
#include "faceDetect.h"
#include <opencv2/opencv.hpp>
#include "DA2Network.hpp"
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

    cv::Mat frame, dstframe, sobelXFrame, sobelYFrame, absSobelX, absSobelY, magnitudeFrame, blurQuantizeFrame, cartoonFrame, embossFrame, faceBlurFrame;
    bool showGrayscale = false;
    bool showAlternativeGreyscale = false;
    bool showSepiaFilter = false;
    bool applyBlur = false;
    bool showSobelX = false;
    bool showSobelY = false;
    bool showMagnitude = false;
    bool showBlurQuantize = false;
    bool faceDetectionEnabled = false;
    bool showCartoonEffect = false;
    bool showEmbossEffect = false;
    bool faceBlurEnabled = false;

    std::cout << "Press 'q' to quit, 's' to save a frame, 'g' to make greyscale, 'h' to make alternative greyscale, 'e' to toggle sepia, 'b' to apply blur, 'x' to show Sobel X, 'y' to show Sobel Y, 'm' to show gradient magnitude, 'l' to show blur and quantize, 'c' to show cartoon effect, 'k' to show emboss effect, 'o' go show blurred face effect.\n";


    while (true) {
        cap >> frame; // Capture a new frame

        if (frame.empty()) {
            std::cout << "Captured empty frame.\n";
            break;
        }

        // Convert the frame to grayscale for face detection
        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // Vector to hold detected faces
        std::vector<cv::Rect> faces;

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

        if (showSepiaFilter) {
            applySepiaTone(frame, dstframe);
        }

        if (applyBlur) {
            blur5x5_2(frame, dstframe); 
            //frame = dstframe.clone();   // Replace original frame with blurred one
        }

        if (showSobelX) {
            sobelX3x3(frame, sobelXFrame);
            cv::convertScaleAbs(sobelXFrame, absSobelX);
            cv::imshow("Video Display", absSobelX);
        } else if (showSobelY) {
            sobelY3x3(frame, sobelYFrame);
            cv::convertScaleAbs(sobelYFrame, absSobelY);
            cv::imshow("Video Display", absSobelY);
        } else if (showMagnitude) {
            sobelX3x3(frame, sobelXFrame);
            sobelY3x3(frame, sobelYFrame);
            magnitude(sobelXFrame, sobelYFrame, magnitudeFrame);
            cv::imshow("Video Display", magnitudeFrame);
        } else if (showBlurQuantize) {
            blurQuantize(frame, blurQuantizeFrame, 10); // Default levels to 10
            cv::imshow("Video Display", blurQuantizeFrame);
        } else if (faceDetectionEnabled) {    // Check if face detection is enabled
            detectFaces(grey, faces);
            drawBoxes(frame, faces, 30, 1.0); // MinWidth: 30 and scale AS 1.0 
            cv::imshow("Video Display", frame);
        } else if (showCartoonEffect) {
            cartoonEffect(frame, cartoonFrame);
            cv::imshow("Video Display", cartoonFrame);
        } else if (showEmbossEffect) {
            embossEffect(frame, embossFrame);
            cv::imshow("Video Display", embossFrame);
        } else if (faceBlurEnabled) {    // Check if face detection is enabled
            detectFaces(grey, faces);
            drawBoxes(frame, faces, 30, 1.0); // MinWidth: 30 and scale AS 1.0 
            applyFaceBlur(frame, faces);
            cv::imshow("Video Display", frame);
        } else {
            cv::imshow("Video Display", frame);
        }


        // Check for key presses
        char key = static_cast<char>(cv::waitKey(10)); // Wait for 10 milliseconds

        if (key == 'q') {
            std::cout << "Quit key pressed. Exiting...\n";
            break;
        } else if (key == 's') {
            static int id = 0;
            std::string outputPath;
            if (showSobelX) {
                outputPath = "saved_sobelX_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, absSobelX)) {
                    std::cout << "Saved Sobel X frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the Sobel X frame.\n";
                }
            } else if (showSobelY) {
                outputPath = "saved_sobelY_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, absSobelY)) {
                    std::cout << "Saved Sobel Y frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the Sobel Y frame.\n";
                }
            } else if (showMagnitude) {
                outputPath = "saved_magnitude_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, magnitudeFrame)) {
                    std::cout << "Saved gradient magnitude frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the gradient magnitude frame.\n";
                }
            } else if (showBlurQuantize) {
                outputPath = "saved_blurquantize_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, blurQuantizeFrame)) {
                    std::cout << "Saved blur quantize frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the blur quantize frame.\n";
                }
            } else if (faceDetectionEnabled) {
                outputPath = "saved_facedetection_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, frame)) {
                    std::cout << "Saved face detection frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the face detection frame.\n";
                }
            } else if (showCartoonEffect) {
                outputPath = "saved_cartoon_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, cartoonFrame)) {
                    std::cout << "Saved cartoon frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the cartoon frame.\n";
                }
            } else if (showEmbossEffect) {
                outputPath = "saved_emboss_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, embossFrame)) {
                    std::cout << "Saved emboss frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the emboss frame.\n";
                }
            } else if (faceBlurEnabled) {
                outputPath = "saved_faceblurred_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, frame)) {
                    std::cout << "Saved face blurred frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the face blurred frame.\n";
                }
            } else {
                outputPath = "saved_frame_" + std::to_string(id++) + ".jpg";
                if (cv::imwrite(outputPath, frame)) {
                    std::cout << "Saved frame as " << outputPath << "\n";
                } else {
                    std::cerr << "Error: Could not save the frame.\n";
                }
            }
        } else if (key == 'g') {
            showGrayscale = !showGrayscale;
            std::cout << "Grayscale mode " << (showGrayscale ? "enabled" : "disabled") << ".\n";
        } else if (key == 'h') {
            showAlternativeGreyscale = !showAlternativeGreyscale;
            std::cout << "Alternative grayscale mode " << (showAlternativeGreyscale ? "enabled" : "disabled") << ".\n";
        } else if (key == 'e') { // sepia
            showSepiaFilter = !showSepiaFilter;
            std::cout << "SepiaFilter mode " << (showSepiaFilter ? "enabled" : "disabled") << ".\n";
        } else if (key == 'b') { // blurred version
            applyBlur = !applyBlur;
            std::cout << "Blur mode " << (applyBlur ? "enabled" : "disabled") << ".\n";
        } else if (key == 'x') { // Sobel X
            showSobelX = !showSobelX;
            showSobelY = false; // Disable Sobel Y if Sobel X is enabled
            std::cout << "Sobel X mode " << (showSobelX ? "enabled" : "disabled") << ".\n";
        } else if (key == 'y') { // Sobel Y
            showSobelY = !showSobelY;
            showSobelX = false; // Disable Sobel X if Sobel Y is enabled
            std::cout << "Sobel Y mode " << (showSobelY ? "enabled" : "disabled") << ".\n";
        } else if (key == 'm') { // Gradient magnitude
            showMagnitude = !showMagnitude;
            showSobelX = false; // Disable Sobel X if magnitude is enabled
            showSobelY = false; // Disable Sobel Y if magnitude is enabled
            std::cout << "Gradient magnitude mode " << (showMagnitude ? "enabled" : "disabled") << ".\n";
        } else if (key == 'l') { // BlurQuantize
            showBlurQuantize = !showBlurQuantize;
            std::cout << "Blur quantize mode " << (showBlurQuantize ? "enabled" : "disabled") << ".\n";
        } else if (key == 'f') {
            faceDetectionEnabled = !faceDetectionEnabled;
            std::cout << "Face detection " << (faceDetectionEnabled ? "enabled" : "disabled") << ".\n";
        } else if (key == 'k') {
            showEmbossEffect = !showEmbossEffect;
            std::cout << "Emboss effect " << (showEmbossEffect ? "enabled" : "disabled") << ".\n";
        } else if (key == 'c') {
            showCartoonEffect = !showCartoonEffect;
            std::cout << "Cartoon effect " << (showCartoonEffect ? "enabled" : "disabled") << ".\n";
        } else if (key == 'o') {
            faceBlurEnabled = !faceBlurEnabled;
            std::cout << "Face blurred effect " << (faceBlurEnabled ? "enabled" : "disabled") << ".\n";
        }
    }

    // Release resources and close windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}