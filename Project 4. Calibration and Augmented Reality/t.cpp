#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open the default camera (camera index 0)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }

    // Chessboard dimensions (inner corners per row and column)
    Size boardSize(9, 6); // 9x6 chessboard (10x7 inner corners)
    float squareSize = 1.0; // Arbitrary world unit per square

    // Lists for storing calibration data
    std::vector<std::vector<Point2f>> corner_list;
    std::vector<std::vector<Vec3f>> point_list;

    // Create a window to display the video feed
    namedWindow("Chessboard Corners", WINDOW_NORMAL);

    vector<Point2f> lastCorners;  // Store the last detected corners
    Mat lastImage;                // Store the last valid frame

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
            // Refine detected corners
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            // Draw corners
            drawChessboardCorners(frame, boardSize, corners, found);

            // Store the last detected corners and frame
            lastCorners = corners;
            lastImage = frame.clone();
        }

        // Display the frame
        imshow("Chessboard Corners", frame);

        // Handle keyboard input
        char key = waitKey(1);
        if (key == 'q') {
            break;
        } else if (key == 's' && !lastCorners.empty()) {
            // Save the detected corners and the corresponding 3D world points
            std::vector<Vec3f> point_set;
            for (int i = 0; i < boardSize.height; i++) {
                for (int j = 0; j < boardSize.width; j++) {
                    point_set.push_back(Vec3f(j * squareSize, -i * squareSize, 0));
                }
            }

            // Append to lists
            corner_list.push_back(lastCorners);
            point_list.push_back(point_set);

            // Save the image
            static int id = 0;
            string outputPath = "saved" + to_string(id++) + ".jpg";
            imwrite(outputPath, lastImage);
            cout << "Image and corners saved!" << endl;
        } else if (key == 'p') {
            // Print stored corner points
            cout << "\n=== Stored Corner Points (2D) ===\n";
            for (size_t i = 0; i < corner_list.size(); i++) {
                cout << "Image " << i << ":\n";
                for (const auto& pt : corner_list[i]) {
                    cout << "(" << pt.x << ", " << pt.y << ") ";
                }
                cout << endl;
            }

            // Print stored world points
            cout << "\n=== Stored World Points (3D) ===\n";
            for (size_t i = 0; i < point_list.size(); i++) {
                cout << "Image " << i << ":\n";
                for (const auto& pt : point_list[i]) {
                    cout << "(" << pt[0] << ", " << pt[1] << ", " << pt[2] << ") ";
                }
                cout << endl;
            }
        }
    }

    // Release resources
    cap.release();
    destroyAllWindows();

    return 0;
}
