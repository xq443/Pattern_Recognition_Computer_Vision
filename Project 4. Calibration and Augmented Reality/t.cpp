/*
  Xujia Qin 
  13th Mar, 2025
  S21
*/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }

    Size boardSize(9, 6);  // Chessboard dimensions
    float squareSize = 1.0;  // Size of each square in arbitrary units

    // Load camera calibration parameters
    Mat camera_matrix, distortion_coefficients;
    FileStorage fs("camera_parameters.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Error: Could not open camera_parameters.xml" << endl;
        return -1;
    }
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistortionCoefficients"] >> distortion_coefficients;
    fs.release();

    namedWindow("Pose Estimation", WINDOW_NORMAL);

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
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            drawChessboardCorners(frame, boardSize, corners, found);

            // Define the 3D points for all corners of the chessboard
            vector<Point3f> objectPoints;
            for (int i = 0; i < boardSize.height; i++) {
                for (int j = 0; j < boardSize.width; j++) {
                    objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));  // Assuming z=0 (flat chessboard)
                }
            }

            // SolvePnP to estimate pose (rotation and translation vectors)
            Mat rvec, tvec;
            solvePnP(objectPoints, corners, camera_matrix, distortion_coefficients, rvec, tvec);

            // Define the 3D points for the four outside corners of the chessboard
            vector<Point3f> targetCorners = {
                Point3f(0, 0, 0),                                  // Top-left
                Point3f((boardSize.width - 1) * squareSize, 0, 0), // Top-right
                Point3f(0, (boardSize.height - 1) * squareSize, 0),// Bottom-left
                Point3f((boardSize.width - 1) * squareSize, (boardSize.height - 1) * squareSize, 0) // Bottom-right
            };

            vector<Point2f> imagePoints;
            // Project the 3D points to 2D
            projectPoints(targetCorners, rvec, tvec, camera_matrix, distortion_coefficients, imagePoints);

            // Draw the projected points
            for (size_t i = 0; i < imagePoints.size(); i++) {
                circle(frame, imagePoints[i], 5, Scalar(0, 0, 255), -1);
            }

            // Draw lines between the projected points
            line(frame, imagePoints[0], imagePoints[1], Scalar(0, 255, 0), 2);
            line(frame, imagePoints[1], imagePoints[3], Scalar(0, 255, 0), 2);
            line(frame, imagePoints[3], imagePoints[2], Scalar(0, 255, 0), 2);
            line(frame, imagePoints[2], imagePoints[0], Scalar(0, 255, 0), 2);
        }

        imshow("Pose Estimation", frame);
        char key = waitKey(1);

        if (key == 'q') {
            break;
        }
        if (key == 's') {
            // Save the current frame with axes
            static int imageId = 0;
            string filename = to_string(imageId++) + ".jpg";
            imwrite(filename, frame);  // Save the image to disk
            cout << "Saved image as " << filename << endl;
        }
    }

    cap.release();
    return 0;
}
