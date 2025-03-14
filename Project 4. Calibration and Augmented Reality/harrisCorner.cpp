/*
  Xujia Qin 
  13th Mar, 2025
  S21
*/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectHarrisCorners(Mat &frame, double threshold) {
    Mat gray, dst, dst_norm, dst_norm_scaled;
    
    // Convert to grayscale
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Harris corner detection
    dst = Mat::zeros(frame.size(), CV_32FC1);
    cornerHarris(gray, dst, 2, 3, 0.04);

    // Normalize the result
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // Draw corners on the frame
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int)dst_norm.at<float>(i, j) > threshold) {
                circle(frame, Point(j, i), 3, Scalar(0, 0, 255), 2);
            }
        }
    }
}

int main() {
    VideoCapture cap(0);  // Open default webcam
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }

    namedWindow("Harris Corner Detection", WINDOW_NORMAL);
    double threshold = 150;  // Initial threshold
    int imageID = 0;  // Initialize image ID

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "Error: Could not capture frame." << endl;
            break;
        }

        detectHarrisCorners(frame, threshold);

        imshow("Harris Corner Detection", frame);
        
        char key = waitKey(1);
        if (key == 'q') break;
        if (key == '+' && threshold < 255) threshold += 5;  // Increase threshold
        if (key == '-' && threshold > 5) threshold -= 5;   // Decrease threshold
        if (key == 's') {
            // Increment the image ID and save the frame with the new filename
            stringstream filename;
            filename << "image_" << imageID++ << ".jpg"; // Create the filename based on the ID

            // Save the current frame as an image
            imwrite(filename.str(), frame);
            cout << "Saved image as " << filename.str() << endl;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
