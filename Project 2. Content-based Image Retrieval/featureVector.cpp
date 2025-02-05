/*
  Xujia Qin 30th Jan, 2025
  S21
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "featureVector.h"
#include "DA2Network.hpp"
#include "utils.h"
#include <cmath>
#include <vector>
#include <string>
#include <cstring> // For strcmp
#include <cstdio>
#include "opencv2/dnn.hpp"     // DNN API include file


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
        for (int c = 0; c < 3; c++) { // Only using first two color channels
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
std::vector<float> twodHistogram(cv::Mat &src, int bins) {
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
  // create a 3D-vector to store the count of colors in rgb color space.
  vector<vector<vector<int> > > hist3d1(
	  bins + 1, vector<vector<int> >(bins + 1, vector<int>(bins + 1, 0)));
  vector<float> result; // store the final normalized 3d-histogram as 1d-feature vector.
  int bin_size = 255/8;
  float total_pixels = 0.0;
  // loop through rows.
  for (int i = 0; i <= src.rows/2; i++) {
	// create row pointer to access rows.
	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  float blue = rptr[j][0];
	  float green = rptr[j][1];
	  float red = rptr[j][2];

	  int blue_index = blue/bin_size;
	  int green_index = green/bin_size;
	  int red_index = red/bin_size;

	  total_pixels++;
	  hist3d1[blue_index][green_index][red_index] += 1;
	}
  }
  // bottom half
  // create a 3D-vector to store the count of colors in rgb color space.
  vector<vector<vector<int> > > hist3d2(
	  bins + 1, vector<vector<int> >(bins + 1, vector<int>(bins + 1)));
  // store the final normalized 3d-histogram as 1d-feature vector.

  // loop through rows.
  for (int i = (src.rows/2) + 1; i < src.rows; i++) {
	// create row pointer to access rows.
	cv::Vec3b *rptr1 = src.ptr<cv::Vec3b>(i);
	for (int j = 0; j < src.cols; j++) {
	  float blue = rptr1[j][0];
	  float green = rptr1[j][1];
	  float red = rptr1[j][2];

	  int blue_index = blue/bin_size;
	  int green_index = green/bin_size;
	  int red_index = red/bin_size;

	  //cout << blue << ":" << blue_index << "," << green << ":" << green_index << "," << red << ":" << red_index << endl;
	  hist3d2[blue_index][green_index][red_index] += 1;
	  total_pixels++;
	}
  }


  // step-2.
  for (int i = 0; i < bins + 1; i++) {
	for (int j = 0; j < bins + 1; j++) {
	  for (int k = 0; k < bins + 1; k++) {
		result.push_back(hist3d1[i][j][k]/total_pixels);
	  }
	}
  }

  for (int i = 0; i < bins + 1; i++) {
	for (int j = 0; j < bins + 1; j++) {
	  for (int k = 0; k < bins + 1; k++) {
		result.push_back(hist3d2[i][j][k]/total_pixels);
	  }
	}
  }
  return result;
}


/*
 * Computes a multi 3D-histogram by taking the gradient magnitude for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
vector<float> colorTexture(cv::Mat &src) {
  // compute a 3d Histogram for whole image.
  vector<float> colorThreeDHist = ThreedHistogram(src);

  // compute a 3d Histogram for the gradient magnitude Image.
  cv::Mat sobelXImg, sobelYImg, gradMagImage;
  sobelX3X3(src, sobelXImg); // get sobelx Image.
  sobelY3X3(src, sobelYImg); // get sobely Image.
  magnitude(sobelXImg, sobelYImg, gradMagImage); // get gradientMag Image.

  vector<float> gradMagThreeDHist = ThreedHistogram(gradMagImage);

  // Merge into single vector
  for (int i = 0; i < gradMagThreeDHist.size(); i++) {
	  colorThreeDHist.push_back(gradMagThreeDHist[i]);
  }

  return colorThreeDHist;
}

// Function to extract HOG features from an image
vector<float> extractHOGFeatures(cv::Mat &src) {
    // Convert to grayscale
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Define HOG descriptor parameters
    cv::HOGDescriptor hog(
        cv::Size(64, 64), // Window size
        cv::Size(16, 16), // Block size
        cv::Size(8, 8),   // Block stride
        cv::Size(8, 8),   // Cell size
        9                 // Number of bins
    );

    vector<float> featureVector;
    hog.compute(gray, featureVector);

    return featureVector;
}


vector<float> LaplaciancolorTexture(cv::Mat &src) {
  vector<float> colorThreeDhist = ThreedHistogram(src); // compute the 3D histogram for the whole image.

  // compute a 3D histogram for the laplacian Image.
  cv::Mat LaplacianImage;
  laplacianFilter(src, LaplacianImage);
  vector<float> LaplacianThreeDHist = ThreedHistogram(LaplacianImage);

  // Merge imto single vector.
  for (int i = 0; i < LaplacianThreeDHist.size(); i++) {
	colorThreeDhist.push_back(LaplacianThreeDHist[i]);
  }

  return colorThreeDhist;
}

// Function to split a string by a delimiter
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to extract the feature vector for a target image
vector<float> extractFeatureVector(const char *targetFilename, const char *featuresFile) {
    vector<float> features;
    ifstream file(featuresFile);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Could not open the features file!" << endl;
        return features;
    }

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        string filename = tokens[0];

        // Check if the current row corresponds to the target image
        if (filename == targetFilename) {
            // Extract the feature values
            for (size_t i = 1; i < tokens.size(); ++i) {
                features.push_back(stof(tokens[i]));
            }
            break; // Stop searching once the target image is found
        }
    }

    file.close();

    if (features.empty()) {
        cerr << "Error: Target image not found in the features file!" << endl;
    }

    return features;
}


// Function to compute a multi histogram for a given image, using depth map to filter pixels
vector<float> depthFilteredMultiHistogram(cv::Mat &src, cv::Mat &depthMap, int bins, float depthThreshold) {
    vector<vector<vector<int>>> hist3d1(
        bins + 1, vector<vector<int>>(bins + 1, vector<int>(bins + 1, 0)));
    vector<float> result;
    int bin_size = 255 / bins;
    float total_pixels = 0.0;

    // Top half
    for (int i = 0; i <= src.rows / 2; i++) {
        cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
        float *dptr = depthMap.ptr<float>(i);
        for (int j = 0; j < src.cols; j++) {
            if (dptr[j] > depthThreshold) continue; // Skip pixels beyond the depth threshold

            float blue = rptr[j][0];
            float green = rptr[j][1];
            float red = rptr[j][2];

            int blue_index = blue / bin_size;
            int green_index = green / bin_size;
            int red_index = red / bin_size;

            total_pixels++;
            hist3d1[blue_index][green_index][red_index] += 1;
        }
    }

    // Bottom half
    vector<vector<vector<int>>> hist3d2(
        bins + 1, vector<vector<int>>(bins + 1, vector<int>(bins + 1, 0)));

    for (int i = (src.rows / 2) + 1; i < src.rows; i++) {
        cv::Vec3b *rptr1 = src.ptr<cv::Vec3b>(i);
        float *dptr1 = depthMap.ptr<float>(i);
        for (int j = 0; j < src.cols; j++) {
            if (dptr1[j] > depthThreshold) continue; // Skip pixels beyond the depth threshold

            float blue = rptr1[j][0];
            float green = rptr1[j][1];
            float red = rptr1[j][2];

            int blue_index = blue / bin_size;
            int green_index = green / bin_size;
            int red_index = red / bin_size;

            hist3d2[blue_index][green_index][red_index] += 1;
            total_pixels++;
        }
    }

    // Normalize and flatten the histograms
    for (int i = 0; i < bins + 1; i++) {
        for (int j = 0; j < bins + 1; j++) {
            for (int k = 0; k < bins + 1; k++) {
                result.push_back(hist3d1[i][j][k] / total_pixels);
            }
        }
    }

    for (int i = 0; i < bins + 1; i++) {
        for (int j = 0; j < bins + 1; j++) {
            for (int k = 0; k < bins + 1; k++) {
                result.push_back(hist3d2[i][j][k] / total_pixels);
            }
        }
    }

    return result;
}

// Function to compute a depth map using the DA2Network
cv::Mat computeDepthMap(cv::Mat &src) {
    cv::Mat dst;
    DA2Network da_net("model_fp16.onnx");

    float scale_factor = 512.0 / (src.rows > src.cols ? src.cols : src.rows);
    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

    da_net.set_input(src, scale_factor);
    da_net.run_network(dst, src.size());

    dst = dst * 5.0; // Scale depth values by 5
    return dst;
}


/*
 * Function to compute 3D Historgram for a given HSV image.
 */

vector<float> HSVHistogram(cv::Mat &src) {
  // Define the number of bins for each channel.
  int hueBins = 30, satBins = 32, ValBins = 32;
  int total_pixels = hueBins*satBins*ValBins;
  vector<float> result;
  for (int i = 0; i < total_pixels; i++) {
	result.push_back(0);
  }

  // Iterate over rows.
  for (int i = 0; i < src.rows; i++) {
	for (int j = 0; j < src.cols; j++) {
	  cv::Vec3b value = src.at<cv::Vec3b>(i, j);
	  // compute the indexes for each channel.
	  float h = value[0];
	  float s = value[1];
	  float v = value[2];

	  int hue_idx = floor((h/180)*hueBins);
	  int sat_idx = floor((s/256)*satBins);
	  int val_idx = floor((v/256)*ValBins);

	  int result_idx = hue_idx*satBins*ValBins + sat_idx*ValBins + val_idx;
	  result[result_idx] += 1;
	}
  }

  // Normalize the histogram.
  for (int i = 0; i < result.size(); i++) {
	result[i] = result[i]/total_pixels;
  }
  return result;
}

/*
 * Thresholds the given Image in HSV format in such a way that, all yellow
 * pixels are whitened.
 */
vector<float> yellowThresholding(cv::Mat &src) {
  int center_row = src.rows/2;
  int center_col = src.cols/2;
  cv::Mat new_src = src(cv::Range(center_row - 50, center_row + 50), cv::Range(center_col - 50, center_col + 50));
  cv::Mat HSVImg;
  cv::cvtColor(new_src, HSVImg, cv::COLOR_BGR2HSV);
  cv::Mat ThresholdImg = cv::Mat::zeros(HSVImg.rows, HSVImg.cols, CV_8UC3);
  vector<float> result;
  // perform thresholding and store it in Thresholding Mat object.
  // Iterate through rows.
  for (int i = 0; i < HSVImg.rows; i++) {
	cv::Vec3b *rptr = HSVImg.ptr<cv::Vec3b>(i);
	cv::Vec3b *dptr = ThresholdImg.ptr<cv::Vec3b>(i);
	// Iterate through cols.
	for (int j = 0; j < HSVImg.cols; j++) {
	  // give the range of yellow color.
	  int lowerHue = 20;
	  int upperHue = 30;
	  int lowerSaturation = 100;
	  int upperSaturation = 255;
	  int lowerValue = 100;
	  int upperValue = 255;

	  int hue = rptr[j][0];
	  int sat = rptr[j][1];
	  int val = rptr[j][2];

	  if ((hue >= lowerHue && hue <= upperHue) && (sat >= lowerSaturation && sat <= upperSaturation)
		  && (val >= lowerValue && val <= upperValue)) {
		dptr[j][0] = 30;
		dptr[j][1] = 254;
		dptr[j][2] = 254;
	  }
	}
  }
  result = HSVHistogram(ThresholdImg);
  return result;
}


vector<float> openCVEmbedding(cv::Mat &src, int debug) {
    const int ORNet_size = 224;

    std::string modelPath = "resnet18-v2-7.onnx";

    // Load the neural network model
    cv::dnn::Net net = cv::dnn::readNet(modelPath);
    cv::Mat blob;

    // Normalize the image to match the model's input requirements
    cv::dnn::blobFromImage(src, blob, (1.0 / 255.0) * (1 / 0.226), cv::Size(ORNet_size, ORNet_size),
                           cv::Scalar(124, 116, 104), true, false, CV_32F);

    net.setInput(blob);
    cv::Mat embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0"); // The name of the embedding layer

    if (debug) {
        cv::imshow("src", src);
        std::cout << embedding << std::endl;
        std::cout << embedding.rows << " " << embedding.cols << std::endl;
        cv::waitKey(0);
    }

    // Convert the Mat to a vector
    std::vector<float> featureVector(embedding.begin<float>(), embedding.end<float>());
    return featureVector;
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
