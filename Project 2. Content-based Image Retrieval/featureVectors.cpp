/*
  Xujia Qin 
  30th Jan, 2025
  S21
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "featureVectors.h"
#include <vector>
#include "utils.h"
#include <cmath>

using namespace std;

/*
 * Use the 7x7 square in the middle of the image as a feature vector. 
 * Use sum-of-squared-difference as the distance metric. 
 */
vector<float> nineXnineSquare(cv::Mat &src) {
  int center_row = src.rows/2;
  int center_col = src.cols/2;
  vector<float> featureVector;

  // create row pointers to rows in Image matrix.
  cv::Vec3b *rptrp4 = src.ptr<cv::Vec3b>(center_row + 4);
  cv::Vec3b *rptrp3 = src.ptr<cv::Vec3b>(center_row + 3);
  cv::Vec3b *rptrp2 = src.ptr<cv::Vec3b>(center_row + 2);
  cv::Vec3b *rptrp1 = src.ptr<cv::Vec3b>(center_row + 1);
  cv::Vec3b *rptr = src.ptr<cv::Vec3b>(center_row);

  cv::Vec3b *rptrm1 = src.ptr<cv::Vec3b>(center_row - 1);
  cv::Vec3b *rptrm2 = src.ptr<cv::Vec3b>(center_row - 2);
  cv::Vec3b *rptrm3 = src.ptr<cv::Vec3b>(center_row - 3);
  cv::Vec3b *rptrm4 = src.ptr<cv::Vec3b>(center_row - 4);

  // loop through rows.
  for (int col = center_col - 3; col <= center_col + 3; col++) {
	// loop through channels
	for (int c = 0; c < 3; c++) {
	  featureVector.push_back(rptr[col][c]);
	  featureVector.push_back(rptrp1[col][c]);
	  featureVector.push_back(rptrp2[col][c]);
	  featureVector.push_back(rptrp3[col][c]);
	  featureVector.push_back(rptrp4[col][c]);

	  featureVector.push_back(rptrm1[col][c]);
	  featureVector.push_back(rptrm2[col][c]);
	  featureVector.push_back(rptrm3[col][c]);
	  featureVector.push_back(rptrm4[col][c]);
	}
  }

  return featureVector;
}

/*
 * Computes the 2D-histogram for a given image.
 * Arg1: src -> source image for which histogram needs to be constructed.
 * Arg2: bins -> Number to bins to quantize.
 */
// vector<float> twodHistogram(cv::Mat &src, int bins) {
//   // create a vector to store the count of colors in rg-chromaticit.
//   int numPixels = src.rows*src.cols;
//   vector<vector<float>> hist2d(bins + 1, vector<float>(bins + 1, 0));
//   vector<float> result;
//   // loop through rows.
//   for (int i = 0; i < src.rows; i++) {
// 	// create a row pointer to access rows in image.
// 	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
// 	// loop through columns.
// 	for (int j = 0; j < src.cols; j++) {
// 	  int total_sum = rptr[j][0] + rptr[j][1] + rptr[j][2];
// 	  // calculte r-value and g-value.
// 	  float r_value = rptr[j][2]/(total_sum + 0.0);
// 	  float g_value = rptr[j][1]/(total_sum + 0.0);

// 	  // find the index to where the count must be increased.
// 	  int r_index = r_value*bins;
// 	  int g_index = g_value*bins;
// 	  //cout << r_index << " " << g_index;
// 	  hist2d[r_index][g_index] += 1;
// 	}
//   }

//   // normalizing the feature vector.
//   for (int i = 1; i < bins + 1; i++) {
// 	for (int j = 1; j < bins + 1; j++) {
// 	  //cout << hist2d[i][j] << endl;
// 	  result.push_back(hist2d[i][j]/numPixels);
// 	}
//   }

//   return result;
// }

// /*
//  * Computes a 3D-histogram for a given image.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */
// vector<float> ThreedHistogram(cv::Mat &src, int bins) {
//   // create a 3D-vector to store the count of colors in rgb color space.
//   vector<vector<vector<float> > > hist3d(
// 	  bins, vector<vector<float> >(bins, vector<float>(bins, 0)));
//   vector<float> result; // store the final normalized 3d-histogram as 1d-feature vector.
//   int bin_size = 256/8;
//   float total_pixels = 0.0;
//   // loop through rows.
//   for (int i = 0; i < src.rows; i++) {
// 	// create row pointer to access rows.
// 	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
// 	for (int j = 0; j < src.cols; j++) {
// 	  float blue = rptr[j][0];
// 	  float green = rptr[j][1];
// 	  float red = rptr[j][2];

// 	  int blue_index = blue/bin_size;
// 	  int green_index = green/bin_size;
// 	  int red_index = red/bin_size;

// 	  total_pixels++;
// 	  hist3d[blue_index][green_index][red_index] += 1;
// 	}
//   }

//   // step-2.
//   for (int i = 0; i < bins; i++) {
// 	for (int j = 0; j < bins; j++) {
// 	  for (int k = 0; k < bins; k++) {
// 		result.push_back(hist3d[i][j][k]/total_pixels);
// 	  }
// 	}
//   }

//   return result;
// }

// /*
//  * Computes a multi 3D-histogram for a given top and bottom half of image.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */

// vector<float> multiHistogram(cv::Mat &src, int bins) {
//   // create a 3D-vector to store the count of colors in rgb color space.
//   vector<vector<vector<int> > > hist3d1(
// 	  bins + 1, vector<vector<int> >(bins + 1, vector<int>(bins + 1, 0)));
//   vector<float> result; // store the final normalized 3d-histogram as 1d-feature vector.
//   int bin_size = 255/8;
//   float total_pixels = 0.0;
//   // loop through rows.
//   for (int i = 0; i <= src.rows/2; i++) {
// 	// create row pointer to access rows.
// 	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
// 	for (int j = 0; j < src.cols; j++) {
// 	  float blue = rptr[j][0];
// 	  float green = rptr[j][1];
// 	  float red = rptr[j][2];

// 	  int blue_index = blue/bin_size;
// 	  int green_index = green/bin_size;
// 	  int red_index = red/bin_size;

// 	  total_pixels++;
// 	  hist3d1[blue_index][green_index][red_index] += 1;
// 	}
//   }
//   // bottom half
//   // create a 3D-vector to store the count of colors in rgb color space.
//   vector<vector<vector<int> > > hist3d2(
// 	  bins + 1, vector<vector<int> >(bins + 1, vector<int>(bins + 1)));
//   // store the final normalized 3d-histogram as 1d-feature vector.

//   // loop through rows.
//   for (int i = (src.rows/2) + 1; i < src.rows; i++) {
// 	// create row pointer to access rows.
// 	cv::Vec3b *rptr1 = src.ptr<cv::Vec3b>(i);
// 	for (int j = 0; j < src.cols; j++) {
// 	  float blue = rptr1[j][0];
// 	  float green = rptr1[j][1];
// 	  float red = rptr1[j][2];

// 	  int blue_index = blue/bin_size;
// 	  int green_index = green/bin_size;
// 	  int red_index = red/bin_size;

// 	  //cout << blue << ":" << blue_index << "," << green << ":" << green_index << "," << red << ":" << red_index << endl;
// 	  hist3d2[blue_index][green_index][red_index] += 1;
// 	  total_pixels++;
// 	}
//   }


//   // step-2.
//   for (int i = 0; i < bins + 1; i++) {
// 	for (int j = 0; j < bins + 1; j++) {
// 	  for (int k = 0; k < bins + 1; k++) {
// 		result.push_back(hist3d1[i][j][k]/total_pixels);
// 	  }
// 	}
//   }

//   for (int i = 0; i < bins + 1; i++) {
// 	for (int j = 0; j < bins + 1; j++) {
// 	  for (int k = 0; k < bins + 1; k++) {
// 		result.push_back(hist3d2[i][j][k]/total_pixels);
// 	  }
// 	}
//   }
//   return result;
// }

// /*
//  * Computes a multi 3D-histogram for a given Image from left and right half of image.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */

// vector<float> multiHistogramLeftRight(cv::Mat &src, int bins) {
//   // create a 3D-vector to store the count of colors in rgb color space.
//   vector<vector<vector<int> > > hist3d1(
// 	  bins + 1, vector<vector<int> >(bins + 1, vector<int>(bins + 1, 0)));
//   vector<float> result; // store the final normalized 3d-histogram as 1d-feature vector.
//   int bin_size = 255/8;
//   float total_pixels = 0.0;
//   // loop through rows.
//   for (int i = 0; i <= src.rows; i++) {
// 	// create row pointer to access rows.
// 	cv::Vec3b *rptr = src.ptr<cv::Vec3b>(i);
// 	for (int j = 0; j < src.cols/2; j++) {
// 	  float blue = rptr[j][0];
// 	  float green = rptr[j][1];
// 	  float red = rptr[j][2];

// 	  int blue_index = blue/bin_size;
// 	  int green_index = green/bin_size;
// 	  int red_index = red/bin_size;

// 	  total_pixels++;
// 	  hist3d1[blue_index][green_index][red_index] += 1;
// 	}
//   }
//   // bottom half
//   // create a 3D-vector to store the count of colors in rgb color space.
//   vector<vector<vector<int> > > hist3d2(
// 	  bins + 1, vector<vector<int> >(bins + 1, vector<int>(bins + 1)));
//   // store the final normalized 3d-histogram as 1d-feature vector.

//   // loop through rows.
//   for (int i = 0; i < src.rows; i++) {
// 	// create row pointer to access rows.
// 	cv::Vec3b *rptr1 = src.ptr<cv::Vec3b>(i);
// 	for (int j = (src.cols/2) + 1; j < src.cols; j++) {
// 	  float blue = rptr1[j][0];
// 	  float green = rptr1[j][1];
// 	  float red = rptr1[j][2];

// 	  int blue_index = blue/bin_size;
// 	  int green_index = green/bin_size;
// 	  int red_index = red/bin_size;

// 	  //cout << blue << ":" << blue_index << "," << green << ":" << green_index << "," << red << ":" << red_index << endl;
// 	  hist3d2[blue_index][green_index][red_index] += 1;
// 	  total_pixels++;
// 	}
//   }


//   // step-2.
//   for (int i = 0; i < bins + 1; i++) {
// 	for (int j = 0; j < bins + 1; j++) {
// 	  for (int k = 0; k < bins + 1; k++) {
// 		result.push_back(hist3d1[i][j][k]/total_pixels);
// 	  }
// 	}
//   }

//   for (int i = 0; i < bins + 1; i++) {
// 	for (int j = 0; j < bins + 1; j++) {
// 	  for (int k = 0; k < bins + 1; k++) {
// 		result.push_back(hist3d2[i][j][k]/total_pixels);
// 	  }
// 	}
//   }
//   return result;
// }

// /*
//  * Computes a multi 3D-histogram by taking the gradient magnitude for a given image.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */
// vector<float> colorTexture(cv::Mat &src) {
//   // compute a 3d Histogram for whole image.
//   vector<float> colorThreeDHist = ThreedHistogram(src);

//   // compute a 3d Histogram for the gradient magnitude Image.
//   cv::Mat sobelXImg, sobelYImg, gradMagImage;
//   sobelX3X3(src, sobelXImg); // get sobelx Image.
//   sobelY3X3(src, sobelYImg); // get sobely Image.
//   magnitude(sobelXImg, sobelYImg, gradMagImage); // get gradientMag Image.

//   vector<float> gradMagThreeDHist = ThreedHistogram(gradMagImage);

//   // Merge into single vector
//   for (int i = 0; i < gradMagThreeDHist.size(); i++) {
// 	colorThreeDHist.push_back(gradMagThreeDHist[i]);
//   }

//   return colorThreeDHist;
// }

// /*
//  * Computes a multi 3D-histogram for a given Image one for image and another by applying gradient magnitude on it.
//  * Arg1: src -> source image for which histogram needs to be constructed.
//  * Arg2: bins -> Number to bins to quantize.
//  */
// vector<float> LaplaciancolorTexture(cv::Mat &src) {
//   vector<float> colorThreeDhist = ThreedHistogram(src); // compute the 3D histogram for the whole image.

//   // compute a 3D histogram for the laplacian Image.
//   cv::Mat LaplacianImage;
//   laplacianFilter(src, LaplacianImage);
//   vector<float> LaplacianThreeDHist = ThreedHistogram(LaplacianImage);

//   // Merge imto single vector.
//   for (int i = 0; i < LaplacianThreeDHist.size(); i++) {
// 	colorThreeDhist.push_back(LaplacianThreeDHist[i]);
//   }

//   return colorThreeDhist;
// }

// /*
//  * Function to compute 3D Historgram for a given HSV image.
//  */

// vector<float> HSVHistogram(cv::Mat &src) {
//   // Define the number of bins for each channel.
//   int hueBins = 30, satBins = 32, ValBins = 32;
//   int total_pixels = hueBins*satBins*ValBins;
//   vector<float> result;
//   for (int i = 0; i < total_pixels; i++) {
// 	result.push_back(0);
//   }

//   // Iterate over rows.
//   for (int i = 0; i < src.rows; i++) {
// 	for (int j = 0; j < src.cols; j++) {
// 	  cv::Vec3b value = src.at<cv::Vec3b>(i, j);
// 	  // compute the indexes for each channel.
// 	  float h = value[0];
// 	  float s = value[1];
// 	  float v = value[2];

// 	  int hue_idx = floor((h/180)*hueBins);
// 	  int sat_idx = floor((s/256)*satBins);
// 	  int val_idx = floor((v/256)*ValBins);

// 	  int result_idx = hue_idx*satBins*ValBins + sat_idx*ValBins + val_idx;
// 	  result[result_idx] += 1;
// 	}
//   }

//   // Normalize the histogram.
//   for (int i = 0; i < result.size(); i++) {
// 	result[i] = result[i]/total_pixels;
//   }
//   return result;
// }

// /*
//  * Thresholds the given Image in HSV format in such a way that, all yellow
//  * pixels are whitened.
//  */
// vector<float> yellowThresholding(cv::Mat &src) {
//   int center_row = src.rows/2;
//   int center_col = src.cols/2;
//   cv::Mat new_src = src(cv::Range(center_row - 50, center_row + 50), cv::Range(center_col - 50, center_col + 50));
//   cv::Mat HSVImg;
//   cv::cvtColor(new_src, HSVImg, cv::COLOR_BGR2HSV);
//   cv::Mat ThresholdImg = cv::Mat::zeros(HSVImg.rows, HSVImg.cols, CV_8UC3);
//   vector<float> result;
//   // perform thresholding and store it in Thresholding Mat object.
//   // Iterate through rows.
//   for (int i = 0; i < HSVImg.rows; i++) {
// 	cv::Vec3b *rptr = HSVImg.ptr<cv::Vec3b>(i);
// 	cv::Vec3b *dptr = ThresholdImg.ptr<cv::Vec3b>(i);
// 	// Iterate through cols.
// 	for (int j = 0; j < HSVImg.cols; j++) {
// 	  // give the range of yellow color.
// 	  int lowerHue = 20;
// 	  int upperHue = 30;
// 	  int lowerSaturation = 100;
// 	  int upperSaturation = 255;
// 	  int lowerValue = 100;
// 	  int upperValue = 255;

// 	  int hue = rptr[j][0];
// 	  int sat = rptr[j][1];
// 	  int val = rptr[j][2];

// 	  if ((hue >= lowerHue && hue <= upperHue) && (sat >= lowerSaturation && sat <= upperSaturation)
// 		  && (val >= lowerValue && val <= upperValue)) {
// 		dptr[j][0] = 30;
// 		dptr[j][1] = 254;
// 		dptr[j][2] = 254;
// 	  }
// 	}
//   }
//   result = HSVHistogram(ThresholdImg);
//   return result;
// }

// /*
//  *  Thresholds the given Image in HSV format in such a way that, all blue
//  * pixels are whitened.
//  */
// vector<float> blueThresholding(cv::Mat &src) {
//   cv::Mat HSVImg;
//   cv::cvtColor(src, HSVImg, cv::COLOR_BGR2HSV);
//   cv::Mat ThresholdImg = cv::Mat::zeros(HSVImg.rows, HSVImg.cols, CV_8UC3);
//   vector<float> result;
//   // perform thresholding and store it in Thresholding Mat object.
//   // Iterate through rows.
//   for (int i = 0; i < HSVImg.rows; i++) {
// 	cv::Vec3b *rptr = HSVImg.ptr<cv::Vec3b>(i);
// 	cv::Vec3b *dptr = ThresholdImg.ptr<cv::Vec3b>(i);
// 	// Iterate through cols.
// 	for (int j = 0; j < HSVImg.cols; j++) {
// 	  // give the range of yellow color.
// 	  int lowerHue = 219;
// 	  int upperHue = 250;
// 	  int lowerSaturation = 92;
// 	  int upperSaturation = 94;
// 	  int lowerValue = 94;
// 	  int upperValue = 97;

// 	  int hue = rptr[j][0];
// 	  int sat = rptr[j][1];
// 	  int val = rptr[j][2];

// 	  if ((hue >= lowerHue && hue <= upperHue) && (sat >= lowerSaturation && sat <= upperSaturation)
// 		  && (val >= lowerValue && val <= upperValue)) {
// 		dptr[j][0] = 30;
// 		dptr[j][1] = 254;
// 		dptr[j][2] = 254;
// 	  }
// 	}
//   }
//   result = HSVHistogram(ThresholdImg);
//   return result;
// }