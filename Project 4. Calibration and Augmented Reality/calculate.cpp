/*
  Xujia Qin 
  13th Mar, 2025
  S21
*/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }

    Size boardSize(9, 6);  // Change this based on the number of squares in your chessboard
    float squareSize = 1.0;  // The real size of a square in your chessboard (in any units)

    Mat camera_matrix, distortion_coefficients;
    FileStorage fs("camera_parameters.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Error: Could not open camera_parameters.xml" << endl;
        return -1;
    }
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistortionCoefficients"] >> distortion_coefficients;
    fs.release();

    cout << "Loaded Camera Matrix:\n" << camera_matrix << endl;
    cout << "Loaded Distortion Coefficients:\n" << distortion_coefficients << endl;

    namedWindow("Pose Estimation", WINDOW_NORMAL);

    int imageID = 0;  // Initialize image ID

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cout << "Error: Could not capture frame." << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(frame, boardSize, corners, found);

            // 3D coordinates of the chessboard corners (object points)
            vector<Point3f> objectPoints;
            for (int i = 0; i < boardSize.height; i++) {
                for (int j = 0; j < boardSize.width; j++) {
                    objectPoints.push_back(Point3f(j * squareSize, -i * squareSize, 0));
                }
            }

            Mat rvec, tvec;
            solvePnP(objectPoints, corners, camera_matrix, distortion_coefficients, rvec, tvec);

            // Projecting 3D points (e.g., chessboard corners) onto the 2D image plane
            vector<Point2f> imagePoints;
            projectPoints(objectPoints, rvec, tvec, camera_matrix, distortion_coefficients, imagePoints);

            // Draw the projected points on the image
            for (size_t i = 0; i < imagePoints.size(); i++) {
                circle(frame, imagePoints[i], 5, Scalar(0, 0, 255), -1); // Draw a red circle at the projected points
            }

            // Alternatively, draw 3D axes at the origin if you want to visualize the 3D axes
            // Define the axes in the 3D space
            vector<Point3f> axesPoints;
            axesPoints.push_back(Point3f(0.0f, 0.0f, 0.0f)); // Origin
            axesPoints.push_back(Point3f(0.1f, 0.0f, 0.0f)); // X-axis
            axesPoints.push_back(Point3f(0.0f, 0.1f, 0.0f)); // Y-axis
            axesPoints.push_back(Point3f(0.0f, 0.0f, 0.1f)); // Z-axis

            vector<Point2f> axesImagePoints;
            projectPoints(axesPoints, rvec, tvec, camera_matrix, distortion_coefficients, axesImagePoints);

            // Draw the axes
            line(frame, axesImagePoints[0], axesImagePoints[1], Scalar(255, 0, 0), 2); // X-axis in blue
            line(frame, axesImagePoints[0], axesImagePoints[2], Scalar(0, 255, 0), 2); // Y-axis in green
            line(frame, axesImagePoints[0], axesImagePoints[3], Scalar(0, 0, 255), 2); // Z-axis in red
        }

        // Show the current frame
        imshow("Pose Estimation", frame);

        char key = waitKey(1);
        if (key == 'q') {
            break;
        } else if (key == 's') {
            // Increment the image ID and save the frame with the new filename
            stringstream filename;
            filename << "image_" << imageID++ << ".jpg"; // Create the filename based on the ID

            // Save the current frame as an image
            imwrite(filename.str(), frame);
            cout << "Saved image as " << filename.str() << endl;
        }
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
