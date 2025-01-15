/*
  Xujia Qin
  January 14th 2025
  CS 5330 OpenCV Project 1 task1
 */

#include <opencv2/opencv.hpp>
#include <cstdio> // gives me printf
#include <cstring> // gives me strcpy

int main(int argc, char *argv[]) {
    cv::Mat src;
    char filename[256];

    // Check for a command-line argument
    if (argc < 2) {
        printf("usage: %s <image filename>\n", argv[0]);
        exit(-1);
    }

    strncpy(filename, argv[1], 255); // Safe copy
    filename[255] = '\0'; // Null-terminate the string

    src = cv::imread(filename); // Load image as BGR
    if(src.data == NULL) { // no data, no image
        printf("error: unable to read image %s\n", filename);
        exit(-2);
    }

    // Display the image initially
    cv::imshow("Image Display", src);
    std::cout << "Press 'q' to quit.\n";

    while (true) {
        int key = cv::waitKey(1);  // Check for key presses with minimal delay

        if (key == 'q') {
            std::cout << "Quit key pressed. Exiting...\n";
            break;
        } 
    }
    
    cv::destroyAllWindows();
    printf("Terminating\n");
    return 0;
}
