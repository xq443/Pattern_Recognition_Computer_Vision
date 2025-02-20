// #include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>
// #include <iostream>
// #include <vector>
// #include <fstream>

// using namespace cv;
// using namespace std;

// int main() {
//     // Load the pre-trained MobileNet SSD model and its configuration
//     string model = "/Users/cathyqin/Desktop/Pattern_Recognition_Computer_Vision/Project 3. Real-time 2-D Object Recognition/frozen_inference_graph.pb"; // Pre-trained model weights
//     string config = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"; // Model configuration
//     cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model, config);

//     // Check if the model was loaded successfully
//     if (net.empty()) {
//         cerr << "Error loading model!" << endl;
//         return -1;
//     }

//     // Load class labels (COCO dataset has 80 classes)
//     vector<string> classes;
//     ifstream ifs("coco.names"); // File containing class names
//     string line;
//     while (getline(ifs, line)) {
//         classes.push_back(line);
//     }

//     // Open the default camera (webcam)
//     VideoCapture cap(0);
//     if (!cap.isOpened()) {
//         cerr << "Error opening video stream or file!" << endl;
//         return -1;
//     }

//     Mat frame;
//     while (true) {
//         // Capture frame-by-frame
//         cap >> frame;
//         if (frame.empty()) break;

//         // Prepare the frame for object detection
//         Mat blob = cv::dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false);
//         net.setInput(blob);

//         // Perform object detection
//         Mat detections = net.forward();

//         // Parse the detections
//         Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
//         for (int i = 0; i < detectionMat.rows; i++) {
//             float confidence = detectionMat.at<float>(i, 2);

//             // Filter out weak detections
//             if (confidence > 0.5) {
//                 int classId = static_cast<int>(detectionMat.at<float>(i, 1));
//                 int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
//                 int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
//                 int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
//                 int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

//                 // Draw bounding box and label
//                 rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
//                 string label = format("%s: %.2f", classes[classId].c_str(), confidence);
//                 putText(frame, label, Point(left, top - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
//             }
//         }

//         // Display the frame with detections
//         imshow("Object Detection", frame);

//         // Exit on 'ESC' key press
//         if (waitKey(1) == 27) break;
//     }

//     // Release resources
//     cap.release();
//     destroyAllWindows();
//     return 0;
// }
