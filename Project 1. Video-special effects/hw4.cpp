#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// Function to calculate the gradient magnitude
Mat computeGradientMagnitude(const Mat& image) {
    // Convert to grayscale if the image is not already
    Mat grayImage;
    if (image.channels() > 1) {
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Calculate gradients using Sobel operator
    Mat gradX, gradY;
    Sobel(grayImage, gradX, CV_64F, 1, 0, 3); // Gradient in x-direction
    Sobel(grayImage, gradY, CV_64F, 0, 1, 3); // Gradient in y-direction

    // Compute gradient magnitude
    Mat gradMagnitude;
    magnitude(gradX, gradY, gradMagnitude);
    return gradMagnitude;
}

// Function to calculate average energy of gradient magnitude
double calculateAverageEnergy(const Mat& gradMagnitude) {
    Scalar sum = cv::sum(gradMagnitude);
    double energy = sum[0] / (gradMagnitude.rows * gradMagnitude.cols);
    return energy;
}

int main() {
    // Load two different textures (images) for analysis
    Mat texture1 = imread("/Users/cathyqin/Desktop/saved_vignette_frame_2.png", IMREAD_COLOR); // Uniform texture
    Mat texture2 = imread("/Users/cathyqin/Desktop/saved_sobelX_frame_1.png", IMREAD_COLOR); // Structured texture (e.g., stripes)ls -l texture1.jpg texture2.jpg
    if (texture1.empty() || texture2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // Calculate gradient magnitude for both textures
    Mat gradMag1 = computeGradientMagnitude(texture1);
    Mat gradMag2 = computeGradientMagnitude(texture2);

    // Calculate average energy for both textures
    double energy1 = calculateAverageEnergy(gradMag1);
    double energy2 = calculateAverageEnergy(gradMag2);

    cout << "Average Energy for Texture 1: " << energy1 << endl;
    cout << "Average Energy for Texture 2: " << energy2 << endl;

    // Display the gradient magnitude images
    imshow("Gradient Magnitude - Texture 1", gradMag1);
    imshow("Gradient Magnitude - Texture 2", gradMag2);

    waitKey(0);
    return 0;
}
