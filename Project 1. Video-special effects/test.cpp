/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1 task1
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

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    cv::VideoWriter video("output_video.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(frame_width, frame_height), true);

    if (!video.isOpened()) {
        std::cerr << "Error: Could not open the output video file.\n";
        return -1;
    }

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
    bool showVignette = false;
    bool showPencilSketch = false;
    bool showOilPainting = false;

    for (int i = 0; i < frame_count; ++i) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame captured.\n";
            break;
        }

        // Convert the frame to grayscale for face detection
        cv::Mat grey;
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // Vector to hold detected faces
        std::vector<cv::Rect> faces;

        if (showGrayscale) {
            greyscale(frame, dstframe);
            frame = dstframe.clone(); // Update frame with grayscale result
            video.write(frame);
        } else if (showAlternativeGreyscale) {
            greyscale(frame, dstframe);
            frame = dstframe.clone(); // Update frame with alternative grayscale result
            video.write(frame);
        } else if (showSepiaFilter) {
            applySepiaTone(frame);
        } else if (applyBlur) {
            if (blur5x5_2(frame, dstframe) == 0) {
                frame = dstframe.clone(); // Update frame with blurred result
                video.write(frame);
            }
        } else if (showSobelX) {
            sobelX3x3(frame, sobelXFrame);
            cv::convertScaleAbs(sobelXFrame, absSobelX);
            cv::imshow("Video Display", absSobelX);
            video.write(absSobelX);
        } else if (showSobelY) {
            sobelY3x3(frame, sobelYFrame);
            cv::convertScaleAbs(sobelYFrame, absSobelY);
            cv::imshow("Video Display", absSobelY);
            video.write(absSobelY);
        } else if (showMagnitude) {
            sobelX3x3(frame, sobelXFrame);
            sobelY3x3(frame, sobelYFrame);
            magnitude(sobelXFrame, sobelYFrame, magnitudeFrame);
            cv::imshow("Video Display", magnitudeFrame);
            video.write(magnitudeFrame);
        } else if (showBlurQuantize) {
            blurQuantize(frame, blurQuantizeFrame, 50); // make it stronger
            cv::imshow("Video Display", blurQuantizeFrame);
            video.write(blurQuantizeFrame);
        } else if (faceDetectionEnabled) {
            detectFaces(grey, faces);
            drawBoxes(frame, faces, 30, 1.0); // MinWidth: 30 and scale AS 1.0 
            cv::imshow("Video Display", frame);
            video.write(frame);
        } else if (faceDetectionEnabled) { // todo
            detectFaces(grey, faces);
            drawBoxes(frame, faces, 30, 1.0); // MinWidth: 30 and scale AS 1.0 
            cv::imshow("Video Display", frame);
            video.write(frame);
        } else if (showCartoonEffect) {
            cartoonEffect(frame, dstframe);
            frame = dstframe.clone(); // Update frame with cartoon effect result
            cv::imshow("Video Display", frame);
            video.write(frame);
        } else if (showEmbossEffect) {
            embossEffect(frame, embossFrame);
            frame = embossFrame.clone(); // Update frame with cartoon effect result
            cv::imshow("Video Display", frame);
            video.write(frame);
        } else if (faceBlurEnabled) {
            detectFaces(grey, faces);
            drawBoxes(frame, faces, 30, 1.0); // MinWidth: 30 and scale AS 1.0 
            applyFaceBlur(frame, faces);
            cv::imshow("Video Display", frame);
        } else if (showVignette) {
            applySepiaToneWithVignette(frame, dstframe);
            cv::imshow("Video Display", dstframe);
            video.write(dstframe);
        } else if (showq) {
            pencilSketch(frame, dstframe);
            cv::imshow("Video Display", frame);
            video.write(dstframe);
        } else if (showOilPainting) {
            oilPainting(frame, dstframe);
            cv::imshow("Video Display", dstframe);
            video.write(dstframe);
        } else {
            video.write(frame);
            cv::imshow("Video Display", frame);
        }

        

        cv::imshow("Video", frame);
        char key = (char)cv::waitKey(1);
        if (key == 'q') {
            std::cout << "Quit key pressed. Exiting...\n";
            break;
        } else if (key == 'g') {
            showGrayscale = !showGrayscale;
            std::cout << "Grayscale mode " << (showGrayscale ? "enabled" : "disabled") << ".\n";
        } else if (key == 'h') {
            showAlternativeGreyscale = !showAlternativeGreyscale;
            std::cout << "Alternative grayscale mode " << (showAlternativeGreyscale ? "enabled" : "disabled") << ".\n";
        } else if (key == 'e') {
            showSepiaFilter = !showSepiaFilter;
            std::cout << "Sepia mode " << (showSepiaFilter ? "enabled" : "disabled") << ".\n";
        } else if (key == 'b') {
            applyBlur = !applyBlur;
            std::cout << "Blur mode " << (applyBlur ? "enabled" : "disabled") << ".\n";
        } else if (key == 'x') {
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
            std::cout << "Magnitude mode " << (showMagnitude ? "enabled" : "disabled") << ".\n";
        } else if (key == 'l') { // Blur and quantize
            showBlurQuantize = !showBlurQuantize;
            std::cout << "Blur and Quantize mode " << (showBlurQuantize ? "enabled" : "disabled") << ".\n";
        } else if (key == 'c') { // Cartoon effect
            showCartoonEffect = !showCartoonEffect;
            std::cout << "Cartoon effect mode " << (showCartoonEffect ? "enabled" : "disabled") << ".\n";
        } else if (key == 'k') { // Emboss effect
            showEmbossEffect = !showEmbossEffect;
            std::cout << "Emboss effect mode " << (showEmbossEffect ? "enabled" : "disabled") << ".\n";
        } else if (key == 'o') { // Face blur
            faceBlurEnabled = !faceBlurEnabled;
            std::cout << "Face blur mode " << (faceBlurEnabled ? "enabled" : "disabled") << ".\n";
        } else if (key == 'v') { // Vignette
            showVignette = !showVignette;
            std::cout << "Vignette effect mode " << (showVignette ? "enabled" : "disabled") << ".\n";
        } else if (key == 't') { // Pencil sketch
            showPencilSketch = !showPencilSketch;
            std::cout << "Pencil Sketch mode " << (showPencilSketch ? "enabled" : "disabled") << ".\n";
        } else if (key == 'u') { // Oil painting
            showOilPainting = !showOilPainting;
            std::cout << "Oil painting effect mode " << (showOilPainting ? "enabled" : "disabled") << ".\n";
        }
    }

    cap.release();
    video.release();
    cv::destroyAllWindows();

    return 0;
}