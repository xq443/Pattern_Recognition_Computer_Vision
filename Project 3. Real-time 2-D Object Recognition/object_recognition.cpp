#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>

using namespace cv;
using namespace std;

// Function to convert BGR to grayscale manually
Mat convertToGray(const Mat &frame) {
    Mat gray(frame.rows, frame.cols, CV_8UC1);
    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            Vec3b pixel = frame.at<Vec3b>(i, j);
            uint8_t grayValue = static_cast<uint8_t>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            gray.at<uint8_t>(i, j) = grayValue;
        }
    }
    return gray;
}

// Function to apply a simple box blur
Mat applyBlur(const Mat &image) {
    Mat blurred(image.rows, image.cols, CV_8UC1);
    int kernel_size = 3;
    int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int sum_kernel = 16;

    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            int sum = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += image.at<uint8_t>(i + ki, j + kj) * kernel[ki + 1][kj + 1];
                }
            }
            blurred.at<uint8_t>(i, j) = sum / sum_kernel;
        }
    }
    return blurred;
}

// Function to compute dynamic threshold using K-means (K=2)
int computeDynamicThreshold(const Mat &gray) {
    vector<int> pixels;
    int sampleRate = 16; // Use 1/16 of pixels for faster computation
    RNG rng;

    for (int i = 0; i < gray.rows; i += sampleRate) {
        for (int j = 0; j < gray.cols; j += sampleRate) {
            pixels.push_back(gray.at<uint8_t>(i, j));
        }
    }

    // K-means clustering with K=2
    int mean1 = pixels[rng.uniform(0, (int)pixels.size())];
    int mean2 = pixels[rng.uniform(0, (int)pixels.size())];

    for (int iter = 0; iter < 5; iter++) { // Iterate a few times for convergence
        vector<int> cluster1, cluster2;
        for (int val : pixels) {
            if (abs(val - mean1) < abs(val - mean2))
                cluster1.push_back(val);
            else
                cluster2.push_back(val);
        }
        if (!cluster1.empty()) mean1 = accumulate(cluster1.begin(), cluster1.end(), 0) / cluster1.size();
        if (!cluster2.empty()) mean2 = accumulate(cluster2.begin(), cluster2.end(), 0) / cluster2.size();
    }

    return (mean1 + mean2) / 2; // Return midpoint as the threshold
}

// Function to threshold an image using a given threshold
Mat thresholdImage(const Mat &gray, int threshold) {
    Mat binary(gray.rows, gray.cols, CV_8UC1);
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            binary.at<uint8_t>(i, j) = (gray.at<uint8_t>(i, j) > threshold) ? 255 : 0;
        }
    }
    return binary;
}

int main() {
    VideoCapture cap(0); // Open webcam
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera!" << endl;
        return -1;
    }

    namedWindow("Original Video", WINDOW_AUTOSIZE);
    namedWindow("Thresholded Video", WINDOW_AUTOSIZE);

    VideoWriter originalVideoWriter("original_video.mp4", VideoWriter::fourcc('a', 'v', 'c', '1'), 30, Size(640, 480));
    VideoWriter thresholdedVideoWriter("thresholded_video.mp4", VideoWriter::fourcc('a', 'v', 'c', '1'), 30, Size(640, 480));

    while (true) {
        Mat frame;
        cap >> frame; // Capture frame
        if (frame.empty()) break;

        Mat gray = convertToGray(frame);
        Mat blurred = applyBlur(gray);
        int threshold = computeDynamicThreshold(blurred);
        Mat binary = thresholdImage(blurred, threshold);

        imshow("Original Video", frame);
        imshow("Thresholded Video", binary);

        // Write the frames to the video files
        originalVideoWriter.write(frame);
        thresholdedVideoWriter.write(binary);

        if (waitKey(1) == 27) break; // Press ESC to exit
    }

    cap.release();
    originalVideoWriter.release();
    thresholdedVideoWriter.release();
    destroyAllWindows();
    return 0;
}
