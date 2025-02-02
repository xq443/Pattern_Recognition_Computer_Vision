/*
  Xujia Qin 30th Jan, 2025
  S21
  identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "featureVector.h"
#include "csv_util.h"
#include "utils.h"
using namespace std;

/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  string featureType = argv[2];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if( argc < 3) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
        strstr(dp->d_name, ".png") ||
        strstr(dp->d_name, ".ppm") ||
        strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);

      vector<float> featureVector;
      // compute the feature vector of each image.
      cv::Mat src = cv::imread(buffer);

      // storing feature vectors in csv file.
      char filename_square_filter[256] =
        "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/featureVectors.csv";
      char filename_Hist2D[256] =
		    "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/hist2DfeatureVector.csv";
	    char filename_Hist3D[256] =
		    "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/hist3DfeatureVector.csv";
	    char filename_multihistogram[256] =
		    "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/multiHistFeaturevectors.csv";
      char filename_texturehistogram[256] =
		  "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/textureHistogram.csv";
      char filename_multihistGaussian[256] =
		  "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 2. Content-based Image Retrieval/laplacianHistFeatures.csv";

      if (featureType=="square") {
        featureVector = sevenXSevenSquare(src); // compute the feature vector.
        append_image_data_csv(filename_square_filter, buffer, featureVector, 0);
      }  else if (featureType=="hist2D") {
        featureVector = twodHistogram(src);
        append_image_data_csv(filename_Hist2D, buffer, featureVector, 0);
      } else if (featureType=="hist3D") {
        featureVector = ThreedHistogram(src);
        append_image_data_csv(filename_Hist3D, buffer, featureVector, 0);
      } else if (featureType=="multiHist") {
        featureVector = multiHistogram(src);
        append_image_data_csv(filename_multihistogram, buffer, featureVector, 0);
      } else if (featureType=="textureHist") {
        featureVector = colorTexture(src);
		    append_image_data_csv(filename_texturehistogram, buffer, featureVector, 0);
      } else if (featureType=="LaplacianHist") {
		    featureVector = LaplaciancolorTexture(src);
		    append_image_data_csv(filename_multihistLaplacian, buffer, featureVector, 0);
      }
  }
  printf("Terminating\n");

  return(0);
}


