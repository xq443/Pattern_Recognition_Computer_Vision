/*
  Xujia Qin 30th Jan, 2025
  S21
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "featureVector.h"
#include <vector>
#include "utils.h"
#include <cmath>

using namespace std;

/*
 * Use the 7x7 square in the middle of the image as a feature vector. 
 * Use sum-of-squared-difference as the distance metric. 
*/


// Function to extract a 7x7 feature vector from the center of an image
vector<float> sevenXSevenSquare(cv::Mat &src) {
    int center_row = src.rows / 2;
    int center_col = src.cols / 2;
    vector<float> featureVector;

    // Ensure image has at least 7x7 pixels
    if (src.rows < 7 || src.cols < 7) {
        cerr << "Error: Image too small for 7x7 extraction!" << endl;
        return featureVector;
    }

    // Row pointers for accessing pixels
    cv::Vec3b *rptrp3 = src.ptr<cv::Vec3b>(center_row + 3);
    cv::Vec3b *rptrp2 = src.ptr<cv::Vec3b>(center_row + 2);
    cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(center_row + 1);
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(center_row);
    cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(center_row - 1);
    cv::Vec3b *rptrm2 = src.ptr<cv::Vec3b>(center_row - 2);
    cv::Vec3b *rptrm3 = src.ptr<cv::Vec3b>(center_row - 3);

    // Loop through the 7x7 window centered at (center_row, center_col)
    for (int col = center_col - 3; col <= center_col + 3; col++) {
        for (int c = 0; c < 2; c++) { // Only using first two color channels
            featureVector.push_back(rptr[col][c]);
            featureVector.push_back(rptrp1[col][c]);
            featureVector.push_back(rptrp2[col][c]);
            featureVector.push_back(rptrp3[col][c]);
            featureVector.push_back(rptrm1[col][c]);
            featureVector.push_back(rptrm2[col][c]);
            featureVector.push_back(rptrm3[col][c]);
        }
    }

    return featureVector;
}

/*
 * Computes the 2D-histogram for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
vector<float> twodHistogram(cv::Mat &src, int bins) {
  // Create a vector to store the count of colors in rg-chromaticity.
  int numPixels = src.rows * src.cols;
  vector<vector<float>> hist2d(bins, vector<float>(bins, 0));
  vector<float> result;

  // Loop through rows.
  for (int i = 0; i < src.rows; i++) {
    // Create a row pointer to access rows in image.
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);

    // Loop through columns.
    for (int j = 0; j < src.cols; j++) {
      int total_sum = rptr[j][0] + rptr[j][1] + rptr[j][2];

      // Avoid division by zero
      if (total_sum == 0) continue;

      // Calculate r-value and g-value.
      float r_value = static_cast<float>(rptr[j][2]) / total_sum;
      float g_value = static_cast<float>(rptr[j][1]) / total_sum;

      // Find the index where the count must be increased.
      int r_index = min(static_cast<int>(r_value * bins), bins - 1);
      int g_index = min(static_cast<int>(g_value * bins), bins - 1);

      hist2d[r_index][g_index] += 1;
    }
  }

  // Flatten 2D histogram into a 1D vector
  for (int i = 0; i < bins; i++) {
    for (int j = 0; j < bins; j++) {
      result.push_back(hist2d[i][j] / numPixels); // Normalize by total pixels
    }
  }

  return result;
}


/*
 * Computes a 3D-histogram for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
vector<float> ThreedHistogram(cv::Mat &src, int bins) {
  // Create a 3D vector to store the count of colors in RGB color space.
  vector<vector<vector<float>>> hist3d(
      bins, vector<vector<float>>(bins, vector<float>(bins, 0)));

  vector<float> result;  // Store the final normalized 3D histogram as a 1D feature vector.
  int bin_size = 256 / bins;  // Compute bin size dynamically.
  float total_pixels = 0.0;

  // Loop through rows.
  for (int i = 0; i < src.rows; i++) {
    // Create a row pointer to access rows.
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      float blue = rptr[j][0];
      float green = rptr[j][1];
      float red = rptr[j][2];

      // Compute indices and ensure they stay within bounds.
      int blue_index = min(static_cast<int>(blue / bin_size), bins - 1);
      int green_index = min(static_cast<int>(green / bin_size), bins - 1);
      int red_index = min(static_cast<int>(red / bin_size), bins - 1);

      total_pixels++;
      hist3d[blue_index][green_index][red_index] += 1;
    }
  }

  // Normalize and flatten the histogram into a 1D vector.
  if (total_pixels > 0) {
    for (int i = 0; i < bins; i++) {
      for (int j = 0; j < bins; j++) {
        for (int k = 0; k < bins; k++) {
          result.push_back(hist3d[i][j][k] / total_pixels);
        }
      }
    }
  } else {
    result.assign(bins * bins * bins, 0.0); // Return a zeroed-out histogram if no pixels were processed.
  }

  return result;
}

/*
 * Computes a multi 3D-histogram for a given top and bottom half of image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */

vector<float> multiHistogram(cv::Mat &src, int bins) {
  // Dynamically compute bin size
  int bin_size = 256 / bins;

  // Create histograms for the top and bottom halves
  vector<vector<vector<int>>> hist3d1(bins, vector<vector<int>>(bins, vector<int>(bins, 0)));
  vector<vector<vector<int>>> hist3d2(bins, vector<vector<int>>(bins, vector<int>(bins, 0)));

  vector<float> result;
  float total_pixels = 0.0;

  // Process the top half of the image
  for (int i = 0; i < src.rows / 2; i++) {
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      float blue = rptr[j][0];
      float green = rptr[j][1];
      float red = rptr[j][2];

      int blue_index = min(static_cast<int>(blue / bin_size), bins - 1);
      int green_index = min(static_cast<int>(green / bin_size), bins - 1);
      int red_index = min(static_cast<int>(red / bin_size), bins - 1);

      hist3d1[blue_index][green_index][red_index] += 1;
      total_pixels++;
    }
  }

  // Process the bottom half of the image
  for (int i = src.rows / 2; i < src.rows; i++) {
    cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      float blue = rptr[j][0];
      float green = rptr[j][1];
      float red = rptr[j][2];

      int blue_index = min(static_cast<int>(blue / bin_size), bins - 1);
      int green_index = min(static_cast<int>(green / bin_size), bins - 1);
      int red_index = min(static_cast<int>(red / bin_size), bins - 1);

      hist3d2[blue_index][green_index][red_index] += 1;
      total_pixels++;
    }
  }

  // Normalize and flatten histograms into a 1D vector
  if (total_pixels > 0) {
    for (int i = 0; i < bins; i++) {
      for (int j = 0; j < bins; j++) {
        for (int k = 0; k < bins; k++) {
          result.push_back(hist3d1[i][j][k] / total_pixels);
        }
      }
    }

    for (int i = 0; i < bins; i++) {
      for (int j = 0; j < bins; j++) {
        for (int k = 0; k < bins; k++) {
          result.push_back(hist3d2[i][j][k] / total_pixels);
        }
      }
    }
  } else {
    result.assign(2 * bins * bins * bins, 0.0);  // Return zeroed histogram if no pixels processed
  }

  return result;
}


// int main() {
//     // Load an image (modify the path as needed)
//     cv::Mat img = cv::imread("olympus/pic.0666.jpg");

//     // Check if the image loaded successfully
//     if (img.empty()) {
//         cerr << "Error: Could not load the image!" << endl;
//         return -1;
//     }

//     // Extract the 7x7 feature vector
//     vector<float> featureVector = sevenXSevenSquare(img);

//     // Check if the feature vector was extracted successfully
//     if (featureVector.empty()) {
//         cerr << "Error: Feature extraction failed!" << endl;
//         return -1;
//     }

//     // Print feature vector values
//     cout << "Feature Vector (size: " << featureVector.size() << "):" << endl;
//     for (size_t i = 0; i < featureVector.size(); i++) {
//         cout << featureVector[i] << " ";
//         if ((i + 1) % 7 == 0) cout << endl; // Format output every 7 elements
//     }

//     return 0;
// }
