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
    RNG rng; // random generator

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

// Function to perform erosion (morphological erosion)
Mat erodeImage(const Mat &binaryImage, int kernelSize) {
    Mat eroded(binaryImage.size(), CV_8UC1, Scalar(0));  // Output image
    
    int k = kernelSize / 2;  // Half of kernel size (for moving window)
    
    // Iterate over each pixel of the image
    for (int i = k; i < binaryImage.rows - k; i++) {
        for (int j = k; j < binaryImage.cols - k; j++) {
            bool erosion = true;
            // Check the neighborhood of the current pixel
            for (int ki = -k; ki <= k; ki++) {
                for (int kj = -k; kj <= k; kj++) {
                    // If any pixel in the neighborhood is 0, set erosion to false
                    if (binaryImage.at<uint8_t>(i + ki, j + kj) == 0) {
                        erosion = false;
                        break;
                    }
                }
                if (!erosion) break;
            }
            // Set the current pixel of the eroded image based on erosion
            eroded.at<uint8_t>(i, j) = erosion ? 255 : 0;
        }
    }
    return eroded;
}

// Function to perform opening (erosion followed by dilation)
Mat openImage(const Mat &binaryImage, int kernelSize) {
    // Perform erosion
    Mat eroded = erodeImage(binaryImage, kernelSize);

    // Perform dilation on the eroded image (reverse of erosion)
    Mat dilated(eroded.size(), CV_8UC1, Scalar(0));  // Output image

    int k = kernelSize / 2;  // Half of kernel size (for moving window)
    
    // Iterate over each pixel of the eroded image
    for (int i = k; i < eroded.rows - k; i++) {
        for (int j = k; j < eroded.cols - k; j++) {
            bool dilation = false;
            // Check the neighborhood of the current pixel
            for (int ki = -k; ki <= k; ki++) {
                for (int kj = -k; kj <= k; kj++) {
                    // If any pixel in the neighborhood is 1, set dilation to true
                    if (eroded.at<uint8_t>(i + ki, j + kj) == 255) {
                        dilation = true;
                        break;
                    }
                }
                if (dilation) break;
            }
            // Set the current pixel of the dilated image based on dilation
            dilated.at<uint8_t>(i, j) = dilation ? 255 : 0;
        }
    }
    return dilated;
}

// Function to perform connected component analysis
Mat analyzeConnectedComponents(const Mat &binaryImage, int minSize = 500) {
    // Run connected components analysis
    Mat labels, stats, centroids;
    int nLabels = connectedComponentsWithStats(binaryImage, labels, stats, centroids, 8, CV_32S);

    // Create an output image for visualization
    Mat output = Mat::zeros(binaryImage.size(), CV_8UC3);

    // Generate random colors for regions
    vector<Vec3b> colors(nLabels);
    RNG rng;
    for (int i = 1; i < nLabels; i++) {
        colors[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    }

    // Loop through each connected component and draw it
    for (int i = 1; i < nLabels; i++) {
        // Region size condition
        if (stats.at<int>(i, CC_STAT_AREA) >= minSize) {
            // Draw the component with random color
            for (int y = 0; y < binaryImage.rows; y++) {
                for (int x = 0; x < binaryImage.cols; x++) {
                    if (labels.at<int>(y, x) == i) {
                        output.at<Vec3b>(y, x) = colors[i];
                    }
                }
            }
        }
    }
    return output;
}

// Function to compute region features (bounding box, centroid, moments)
void computeRegionFeatures(const Mat &regionMap, int regionID, const string& imageName) {
    // Find connected components and stats
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(regionMap, labels, stats, centroids);

    // Extract stats for the given region
    int x = stats.at<int>(regionID, CC_STAT_LEFT);
    int y = stats.at<int>(regionID, CC_STAT_TOP);
    int width = stats.at<int>(regionID, CC_STAT_WIDTH);
    int height = stats.at<int>(regionID, CC_STAT_HEIGHT);
    int area = stats.at<int>(regionID, CC_STAT_AREA);

    // Extract centroid
    double cx = centroids.at<double>(regionID, 0);
    double cy = centroids.at<double>(regionID, 1);

    // Create a mask for the region
    Mat regionMask = (labels == regionID);

    // Compute moments
    Moments moment = moments(regionMask, false);
    
    // Calculate the central moments (mu20, mu02, mu11)
    double mu20 = moment.mu20;
    double mu02 = moment.mu02;
    double mu11 = moment.mu11;
    
    // Calculate the angle of the axis of least central moment
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
    
    // Bounding box aspect ratio (width / height)
    double boundingBoxRatio = static_cast<double>(width) / height;

    // Percent filled (area vs bounding box area)
    double boundingBoxArea = static_cast<double>(width) * height;
    double percentFilled = static_cast<double>(area) / boundingBoxArea;

    // Print out the region features
    cout << "Region " << regionID << " features:" << endl;
    cout << "  Centroid: (" << cx << ", " << cy << ")" << endl;
    cout << "  Bounding Box: " << width << "x" << height << endl;
    cout << "  Bounding Box Ratio: " << boundingBoxRatio << endl;
    cout << "  Percent Filled: " << percentFilled << endl;
    cout << "  Angle (axis of least central moment): " << theta * 180 / CV_PI << " degrees" << endl;

    // Draw bounding box in RED with THICKER edges
    Mat outputImage;
    cvtColor(regionMap, outputImage, COLOR_GRAY2BGR);  // Convert grayscale to BGR for color drawing
    rectangle(outputImage, Point(x, y), Point(x + width, y + height), Scalar(0, 0, 255), 3); // Red bounding box

    // Draw centroid in GREEN
    circle(outputImage, Point(static_cast<int>(cx), static_cast<int>(cy)), 5, Scalar(0, 255, 0), -1);
    
    // Draw the orientation axis in BLUE with THICKER lines and EXTENDED length
    int axisLength = max(width, height) / 2; // Make the axis proportionate to the region size
    int endX1 = static_cast<int>(cx + axisLength * cos(theta));
    int endY1 = static_cast<int>(cy - axisLength * sin(theta));
    int endX2 = static_cast<int>(cx - axisLength * cos(theta));
    int endY2 = static_cast<int>(cy + axisLength * sin(theta));
    line(outputImage, Point(endX1, endY1), Point(endX2, endY2), Scalar(255, 0, 0), 3); // Blue axis line

    // Save the output image as a PNG
    imwrite(imageName, outputImage);  // Saving as PNG
    
    // Show the result
    imshow("Region " + to_string(regionID), outputImage);
    
    waitKey(0);
    destroyAllWindows();
}


int main() {
    // Load image instead of capturing from webcam
    Mat frame = imread("/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 3. Real-time 2-D Object Recognition/db/f2.png"); // Change "input.png" to your image file
    if (frame.empty()) {
        cerr << "Error: Cannot open image file!" << endl;
        return -1;
    }

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Thresholded Image", WINDOW_AUTOSIZE);
    namedWindow("Opened Image", WINDOW_AUTOSIZE);
    namedWindow("Region Map", WINDOW_AUTOSIZE);

    Mat gray = convertToGray(frame);
    Mat blurred = applyBlur(gray);
    int threshold = computeDynamicThreshold(blurred);
    Mat binary = thresholdImage(blurred, threshold);

    // Apply morphological filtering (Opening)
    Mat cleaned = openImage(binary, 3);

    // Perform connected components analysis
    Mat regionMap = analyzeConnectedComponents(cleaned, 500);  // Minimum size set to 500

    // imshow("Original Image", frame);
    // imshow("Thresholded Image", binary);
    // imshow("Opened Image", cleaned);
    // imshow("Region Map", regionMap);

    // // Save processed images
    // imwrite("1_original.png", frame);
    // imwrite("2_thresholded.png", binary);
    // imwrite("3_opened.png", cleaned);
    // imwrite("4_region_map.png", regionMap);
    // cout << "All images saved successfully." << endl;

    // Find and display regions
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labels, stats, centroids);
    for (int i = 1; i < min(numLabels, 20); i++) { // Iterate over detected regions
        string imageName = "region_" + to_string(i) + ".png";
        computeRegionFeatures(binary, i, imageName);  // Pass PNG filename to function
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}
