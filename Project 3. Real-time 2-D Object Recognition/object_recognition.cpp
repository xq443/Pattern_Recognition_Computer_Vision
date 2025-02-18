#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
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

// Function to compute region features (bounding box, centroid, moments)
vector<double> computeRegionFeatures(const Mat &regionMap, int regionID) {
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

    // Create the feature vector
    vector<double> featureVector = {cx, cy, boundingBoxRatio, percentFilled, theta * 180 / CV_PI}; // Angle in degrees
    return featureVector;
}

// Function to store the feature vector and label in a file
void storeFeatureVector(const vector<double> &featureVector, const string &label, const string &filename = "object_db.csv") {
    ofstream outFile(filename, ios::app); // Open in append mode
    if (outFile.is_open()) {
        outFile << label;
        for (double feature : featureVector) {
            outFile << "," << feature;
        }
        outFile << endl;
        outFile.close();
    } else {
        cerr << "Error opening file for writing!" << endl;
    }
}

// Function to handle the training mode (labeling objects)
void trainObject(Mat &frame, const Mat &binary) {
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labels, stats, centroids);

    // Display the regions in the frame
    for (int i = 1; i < numLabels; i++) { // Skip background
        vector<double> featureVector = computeRegionFeatures(binary, i);

        // Display the region and ask for user input
        cout << "Object detected, please enter a label for it: ";
        string label;
        cin >> label;

        // Store the feature vector and label
        storeFeatureVector(featureVector, label);

        // Visualize the region on the frame
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        rectangle(frame, Point(x, y), Point(x + width, y + height), Scalar(255, 0, 0), 2);
        putText(frame, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
    }

    // Display the updated frame
    imshow("Training Mode", frame);
}

// Function to compute the Euclidean distance with scaled features
double computeEuclideanDistance(const vector<double>& feature1, const vector<double>& feature2, const vector<double>& stdev) {
    double distance = 0.0;
    for (size_t i = 0; i < feature1.size(); i++) {
        // Scale the feature by its standard deviation
        double diff = (feature1[i] - feature2[i]) / stdev[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

// Function to load the object database from a CSV file
void loadObjectDatabase(const string& filename, vector<vector<double>>& featureVectors, vector<string>& labels) {
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string label;
        getline(ss, label, ',');
        labels.push_back(label);
        
        vector<double> features;
        string feature;
        while (getline(ss, feature, ',')) {
            features.push_back(stod(feature));
        }
        featureVectors.push_back(features);
    }
}

// Function to calculate the standard deviation for each feature in the database
vector<double> calculateStandardDeviations(const vector<vector<double>>& featureVectors) {
    size_t numFeatures = featureVectors[0].size();
    vector<double> stdev(numFeatures, 0.0);
    size_t numVectors = featureVectors.size();
    
    // Calculate mean for each feature
    vector<double> means(numFeatures, 0.0);
    for (const auto& features : featureVectors) {
        for (size_t i = 0; i < numFeatures; i++) {
            means[i] += features[i];
        }
    }
    for (size_t i = 0; i < numFeatures; i++) {
        means[i] /= numVectors;
    }

    // Calculate standard deviation
    for (const auto& features : featureVectors) {
        for (size_t i = 0; i < numFeatures; i++) {
            stdev[i] += pow(features[i] - means[i], 2);
        }
    }
    for (size_t i = 0; i < numFeatures; i++) {
        stdev[i] = sqrt(stdev[i] / numVectors);
    }
    
    return stdev;
}

// Function to classify an unknown object by finding the closest feature vector in the object DB
string classifyObject(const vector<double>& unknownFeatureVector, const vector<vector<double>>& featureVectors, const vector<string>& labels, const vector<double>& stdev) {
    double minDistance = numeric_limits<double>::infinity();
    string closestLabel;
    
    for (size_t i = 0; i < featureVectors.size(); i++) {
        double dist = computeEuclideanDistance(unknownFeatureVector, featureVectors[i], stdev);
        if (dist < minDistance) {
            minDistance = dist;
            closestLabel = labels[i];
        }
    }
    return closestLabel;
}

// Function to handle real-time classification and display the label on the output
void classifyAndDisplay(Mat& frame, const Mat& binary, const vector<vector<double>>& featureVectors, const vector<string>& labels, const vector<double>& stdev) {
    Mat labelsMat, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labelsMat, stats, centroids);

    // Display the regions in the frame
    for (int i = 1; i < numLabels; i++) { // Skip background
        vector<double> featureVector = computeRegionFeatures(binary, i);
        
        // Classify the unknown object
        string label = classifyObject(featureVector, featureVectors, labels, stdev);

        // Visualize the region on the frame with its label
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);

        rectangle(frame, Point(x, y), Point(x + width, y + height), Scalar(255, 0, 0), 2);
        putText(frame, label, Point(x, y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
    }

    // Display the updated frame with the classification
    imshow("Classified Video", frame);
}

int main() {
    // Load the object database from file
    vector<vector<double>> featureVectors;
    vector<string> labels;
    loadObjectDatabase("object_db.csv", featureVectors, labels);

    // Calculate the standard deviations for feature normalization
    vector<double> stdev = calculateStandardDeviations(featureVectors);

    VideoCapture cap(0); // Open webcam
    if (!cap.isOpened()) {
        cerr << "Error: Cannot open camera!" << endl;
        return -1;
    }

    namedWindow("Original Video", WINDOW_AUTOSIZE / 2);
    namedWindow("Thresholded Video", WINDOW_AUTOSIZE / 2);

    while (true) {
        Mat frame;
        cap >> frame; // Capture a frame from the video feed
        if (frame.empty()) {
            cerr << "Error: Failed to capture frame!" << endl;
            break;
        }

        Mat gray = convertToGray(frame); // Convert to grayscale
        int dynamicThreshold = computeDynamicThreshold(gray); // Compute dynamic threshold
        Mat binary = thresholdImage(gray, dynamicThreshold); // Threshold the image

        // Handle the classification and display
        classifyAndDisplay(frame, binary, featureVectors, labels, stdev);

        // Show the current frame and thresholded frame
        imshow("Original Video", frame);
        imshow("Thresholded Video", binary);

        char key = (char)waitKey(1); // Wait for user input
        if (key == 's' || key == 'S') {
            // Save the current frame as a PNG image
            static int frameCount = 0; // To avoid overwriting files
            string filename1 = "frame_" + to_string(frameCount++) + ".png";
            string filename2 = "frame_" + to_string(frameCount++) + ".png";
            imwrite(filename1, frame);
            imwrite(filename2, binary);
            cout << "Saved current frame as " << filename << endl;
        } else if (key == 'q' || key == 'Q') {
            // Quit the program
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
