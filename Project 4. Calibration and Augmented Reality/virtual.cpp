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
    float squareSize = 1.0;  // Each square is 1 unit

    // Load camera parameters
    Mat camera_matrix, distortion_coefficients;
    FileStorage fs("camera_parameters.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Error: Could not open camera parameters file." << endl;
        return -1;
    }
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistortionCoefficients"] >> distortion_coefficients;
    fs.release();

    namedWindow("Virtual Object Projection", WINDOW_NORMAL);

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

            // Define the 3D coordinates of all chessboard corners
            vector<Point3f> objectPoints;
            for (int i = 0; i < boardSize.height; i++) {
                for (int j = 0; j < boardSize.width; j++) {
                    objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
                }
            }

            // Estimate pose using solvePnP
            Mat rvec, tvec;
            solvePnP(objectPoints, corners, camera_matrix, distortion_coefficients, rvec, tvec);

            // Define structure
            vector<Point3f> houseModel = {
                {1, 1, 0}, {4, 1, 0}, {4, 4, 0}, {1, 4, 0}, // Base square
                {1, 1, 3}, {4, 1, 3}, {4, 4, 3}, {1, 4, 3}, // Top of walls
                {2.5, 0.5, 5}, {2.5, 4.5, 5}  // Roof peak 
            };

            // Project 3D object to 2D
            vector<Point2f> projectedPoints;
            projectPoints(houseModel, rvec, tvec, camera_matrix, distortion_coefficients, projectedPoints);

            // Draw house base
            for (int i = 0; i < 4; i++) {
                line(frame, projectedPoints[i], projectedPoints[(i + 1) % 4], Scalar(255, 0, 0), 2);
                line(frame, projectedPoints[i + 4], projectedPoints[((i + 1) % 4) + 4], Scalar(255, 0, 0), 2);
                line(frame, projectedPoints[i], projectedPoints[i + 4], Scalar(0, 255, 0), 2);
            }

            // Draw roof
            line(frame, projectedPoints[4], projectedPoints[8], Scalar(0, 0, 255), 2);
            line(frame, projectedPoints[5], projectedPoints[8], Scalar(0, 0, 255), 2);
            line(frame, projectedPoints[6], projectedPoints[9], Scalar(0, 0, 255), 2);
            line(frame, projectedPoints[7], projectedPoints[9], Scalar(0, 0, 255), 2);
            line(frame, projectedPoints[8], projectedPoints[9], Scalar(0, 0, 255), 2);
        }

        imshow("Virtual Object Projection", frame);
        char key = waitKey(1);

        if (key == 'q') {
            break;
        }
        if (key == 's') {
            static int imageId = 0;
            string filename = to_string(imageId++) + ".jpg";
            imwrite(filename, frame);
            cout << "Saved image as " << filename << endl;
        }
    }

    cap.release();
    return 0;
}
