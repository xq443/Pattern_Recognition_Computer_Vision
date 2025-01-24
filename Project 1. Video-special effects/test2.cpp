#include <opencv2/opencv.hpp>
#include "faceDetect.h"
#include <vector>

int main(int argc, char** argv) {
    // Check if video file path or webcam index is provided
    if (argc < 3) {
        printf("Usage: %s <input_video_or_camera_index> <output_video>\n", argv[0]);
        return -1;
    }

    // Open video file or webcam
    cv::VideoCapture cap;
    if (isdigit(argv[1][0])) {
        int camIndex = std::stoi(argv[1]);
        cap.open(camIndex);
    } else {
        cap.open(argv[1]);
    }

    if (!cap.isOpened()) {
        printf("Error: Unable to open video source\n");
        return -1;
    }

    // Get the video frame properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Create a VideoWriter object to save the output
    cv::VideoWriter writer;
    writer.open(argv[2], cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(frameWidth, frameHeight));

    if (!writer.isOpened()) {
        printf("Error: Unable to open output video file for writing\n");
        return -1;
    }

    // Window for displaying the video
    cv::namedWindow("Face Detection with Blur", cv::WINDOW_AUTOSIZE);

    cv::Mat frame, grey;
    std::vector<cv::Rect> faces;

    while (true) {
        // Capture each frame
        cap >> frame;
        if (frame.empty()) break; // Stop if the video ends

        // Convert to grayscale for face detection
        cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);

        // Detect faces in the frame
        detectFaces(grey, faces);

        // Apply face blur to detected regions
        // applyFaceBlur(frame, faces);

        // Write the processed frame to the output video
        writer.write(frame);

        // Display the frame with blurred faces
        cv::imshow("Face Detection with Blur", frame);

        // Break the loop on pressing 'q'
        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
