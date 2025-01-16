/*
  Xujia Qin
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/
#include "timeBlur.h"
#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

// prototypes for the functions to test
int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );

// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}


int blur5x5_1( cv::Mat &src, cv::Mat &dst ) {
  dst = src;

  // Gaussian
  int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
  };

  int sum = 100;

  // Loop through the image, skipping the outer two rows and columns
    for (int row = 2; row < src.rows - 2; ++row) {
        for (int col = 2; col < src.cols - 2; ++col) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply the 5x5 kernel
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(row + i, col + j);
                    int weight = kernel[i + 2][j + 2];
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // Normalize and clamp the values to the 0-255 range
            uchar newB = std::min(255, std::max(0, sumB / sum));
            uchar newG = std::min(255, std::max(0, sumG / sum));
            uchar newR = std::min(255, std::max(0, sumR / sum));

            dst.at<cv::Vec3b>(row, col) = cv::Vec3b(newB, newG, newR);
        }
    }

    return 0; // success
}


int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    dst = src;
    
    // Separable kernel (1D kernels)
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; 
    
    // Horizontal pass (row-wise)
    for (int row = 0; row < src.rows; ++row) {
        // Get pointers to the current row
        cv::Vec3b* rowPtr = src.ptr<cv::Vec3b>(row);
        cv::Vec3b* dstPtr = dst.ptr<cv::Vec3b>(row);
        
        for (int col = 2; col < src.cols - 2; ++col) {
            int sumB = 0, sumG = 0, sumR = 0;
            
            // Convolve the kernel along the row (horizontal)
            for (int k = -2; k <= 2; ++k) {
                cv::Vec3b pixel = rowPtr[col + k];  // Access pixel using row pointer
                int weight = kernel[k + 2]; // Get the corresponding kernel weight
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            // Store the result back in the destination image
            dstPtr[col] = cv::Vec3b(
                std::min(255, std::max(0, sumB / kernelSum)),
                std::min(255, std::max(0, sumG / kernelSum)),
                std::min(255, std::max(0, sumR / kernelSum))
            );
        }
    }

    // Vertical pass (column-wise)
    cv::Mat temp = dst; // Temporary image to store intermediate results
    for (int col = 0; col < src.cols; ++col) {
        for (int row = 2; row < src.rows - 2; ++row) {
            int sumB = 0, sumG = 0, sumR = 0;
            
            // Get pointers to the current column
            cv::Vec3b* tempPtr = temp.ptr<cv::Vec3b>(row);
            cv::Vec3b* dstPtr = dst.ptr<cv::Vec3b>(row);
            
            // Convolve the kernel along the column (vertical)
            for (int k = -2; k <= 2; ++k) {
                cv::Vec3b pixel = tempPtr[row + k];  // Access pixel using row pointer
                int weight = kernel[k + 2]; // Get the corresponding kernel weight
                sumB += pixel[0] * weight;
                sumG += pixel[1] * weight;
                sumR += pixel[2] * weight;
            }

            // Store the result back in the destination image
            dstPtr[col] = cv::Vec3b(
                std::min(255, std::max(0, sumB / kernelSum)),
                std::min(255, std::max(0, sumG / kernelSum)),
                std::min(255, std::max(0, sumR / kernelSum))
            );
        }
    }

    return 0; // success
}

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

// int main(int argc, char *argv[]) {  // main function, execution starts here
//   cv::Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
//   cv::Mat dst; // cv::Mat to hold the output of the process
//   char filename[256]; // a string for the filename

//   // usage: checking if the user provided a filename
//   if(argc < 2) {
//     printf("Usage %s <image filename>\n", argv[0]);
//     exit(-1);
//   }
//   strcpy(filename, argv[1]); // copying 2nd command line argument to filename variable

//   // read the image
//   src = cv::imread(filename); // allocating the image data
//   // test if the read was successful
//   if(src.data == NULL) {  // src.data is the reference to the image data
//     printf("Unable to read image %s\n", filename);
//     exit(-1);
//   }

//   const int Ntimes = 10;
	
//   //////////////////////////////
//   // set up the timing for version 1
//   double startTime = getTime();

//   // execute the file on the original image a couple of times
//   for(int i=0;i<Ntimes;i++) {
//     blur5x5_1( src, dst );
//   }

//   // end the timing
//   double endTime = getTime();

//   // compute the time per image
//   double difference = (endTime - startTime) / Ntimes;

//   // print the results
//   printf("Time per image (1): %.4lf seconds\n", difference );

//   //////////////////////////////
//   // set up the timing for version 2
//   startTime = getTime();

//   // execute the file on the original image a couple of times
//   for(int i=0;i<Ntimes;i++) {
//     blur5x5_2( src, dst );
//   }

//   // end the timing
//   endTime = getTime();

//   // compute the time per image
//   difference = (endTime - startTime) / Ntimes;

//   // print the results
//   printf("Time per image (2): %.4lf seconds\n", difference );
  
//   // terminate the program
//   printf("Terminating\n");

//   return(0);
// }
