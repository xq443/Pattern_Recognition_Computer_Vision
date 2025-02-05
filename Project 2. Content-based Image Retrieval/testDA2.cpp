#include <opencv2/opencv.hpp>
#include <iostream>
#include "DA2Network.hpp"
using namespace std;

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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image filename>\n";
        return -1;
    }

    // Load the input image
    cv::Mat src = cv::imread(argv[1]);
    if (src.empty()) {
        std::cerr << "Error: Unable to read image " << argv[1] << std::endl;
        return -1;
    }

    // Compute depth map
    cv::Mat depthMap = computeDepthMap(src);

    // Normalize depth map for visualization
    cv::Mat depthVis;
    cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply a colormap for better visualization
    cv::applyColorMap(depthVis, depthVis, cv::COLORMAP_INFERNO);

    // Display the original image and depth map
    cv::imshow("Original Image", src);
    cv::imshow("Depth Map", depthVis);

    // Save depth image
    cv::imwrite("depth_output.png", depthVis);
    
    std::cout << "Depth map saved as depth_output.png" << std::endl;

    cv::waitKey(0);
    return 0;
}
