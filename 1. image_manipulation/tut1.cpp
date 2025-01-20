/*
  Bruce A. Maxwell
  January 2025
  CS 5330 OpenCV tutorial
 */

#include <cstdio> // gives me printf
#include <cstring> // gives me strcpy
#include <opencv2/opencv.hpp> // openCV

int main(int argc, char *argv[]) {
  cv::Mat src;
  cv::Mat dst;
  char filename[256];

  // check for a command line argument
  if(argc < 2 ) {
    printf("usage: %s  <image filename>\n", argv[0]); // argv[0] is the program name
    exit(-1);
  }
  strncpy(filename, argv[1], 255); // safe strcpy

  src = cv::imread( filename ); // by default, returns image as 8-bit BGR image (if it's color), use IMREAD_UNCHANGED to keep the original data format
  if( src.data == NULL) { // no data, no image
    printf("error: unable to read image %s\n", filename);
    exit(-2);
  }

  cv::imshow( filename, src ); // display the original image

  // modify the image
  src.copyTo( dst ); // copy the src image to dst image

  /*
  // use the at<> method to modify pixels
  for(int i=0;i<dst.rows;i++) {
    for(int j=0;j<dst.cols;j++) {
      // swap the red and blue channels
      unsigned char tmp = dst.at<cv::Vec3b>(i, j)[0]; // get the blue channel
      dst.at<cv::Vec3b>(i, j)[0] = dst.at<cv::Vec3b>(i, j)[2]; // assign the red value to the blue channel
      dst.at<cv::Vec3b>(i, j)[2] = tmp; // assign the blue value to the red channel
    }
  }
  */

  // use the at<> method to modify pixels with a box filter
  // 1 1 1
  // 1 1 1
  // 1 1 1
  for(int i=1;i<dst.rows-1;i++) {
    for(int j=1;j<dst.cols-1;j++) {
      for(int k=0;k<3;k++) {
	dst.at<cv::Vec3b>(i, j)[k] = (src.at<cv::Vec3b>(i-1, j-1)[k] + src.at<cv::Vec3b>(i-1, j)[k] + src.at<cv::Vec3b>(i-1, j+1)[k]
				      + src.at<cv::Vec3b>(i, j-1)[k] + src.at<cv::Vec3b>(i, j)[k] + src.at<cv::Vec3b>(i, j+1)[k]
				      + src.at<cv::Vec3b>(i+1, j-1)[k] + src.at<cv::Vec3b>(i+1, j)[k] + src.at<cv::Vec3b>(i+1, j+1)[k]) / 9;
      }
    }
  }
  
  /*
  // use the ptr<> method, this is much, much faster 
  for(int i=0;i<dst.rows;i++) {
    cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i); // get the pointer for the row i data
    for(int j=0;j<dst.cols;j++) {
      unsigned char tmp = ptr[j][0];
      ptr[j][0] = ptr[j][2];
      ptr[j][2] = tmp;
    }
  }
  */

  cv::imshow( "swapped", dst );

  cv::waitKey(0); // blocking call with an argument of 0

  cv::imwrite( "swap.png", dst );

  printf("Terminating\n");
  
  return(0);
}

  
  

