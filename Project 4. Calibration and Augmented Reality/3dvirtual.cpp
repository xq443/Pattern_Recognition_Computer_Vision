#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

using namespace cv;
using namespace std;

// OpenGL cube vertices and colors
GLfloat cubeVertices[] = {
    -0.5f, -0.5f, -0.5f,   0.5f, -0.5f, -0.5f,   0.5f,  0.5f, -0.5f,  -0.5f,  0.5f, -0.5f,
    -0.5f, -0.5f,  0.5f,   0.5f, -0.5f,  0.5f,   0.5f,  0.5f,  0.5f,  -0.5f,  0.5f,  0.5f
};
GLuint cubeIndices[] = {
    0, 1, 2, 2, 3, 0,  1, 5, 6, 6, 2, 1,
    7, 6, 5, 5, 4, 7,  4, 0, 3, 3, 7, 4,
    4, 5, 1, 1, 0, 4,  3, 2, 6, 6, 7, 3
};


void renderCube(Mat &cameraMatrix, Mat &distCoeffs, Mat &rvec, Mat &tvec) {
    // Convert rotation vector to matrix
    Mat rotMat;
    Rodrigues(rvec, rotMat);

    // Construct OpenGL model-view matrix
    GLfloat modelView[16] = {
        static_cast<GLfloat>(rotMat.at<double>(0,0)), static_cast<GLfloat>(rotMat.at<double>(0,1)), static_cast<GLfloat>(rotMat.at<double>(0,2)), 0.0f,
        static_cast<GLfloat>(rotMat.at<double>(1,0)), static_cast<GLfloat>(rotMat.at<double>(1,1)), static_cast<GLfloat>(rotMat.at<double>(1,2)), 0.0f,
        static_cast<GLfloat>(rotMat.at<double>(2,0)), static_cast<GLfloat>(rotMat.at<double>(2,1)), static_cast<GLfloat>(rotMat.at<double>(2,2)), 0.0f,
        static_cast<GLfloat>(tvec.at<double>(0)), static_cast<GLfloat>(tvec.at<double>(1)), static_cast<GLfloat>(tvec.at<double>(2)), 1.0f
    };

    // Apply transformations
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(modelView);

    // Draw the cube
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, cubeVertices);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, cubeIndices);
    glDisableClientState(GL_VERTEX_ARRAY);
}

int main() {
    // Initialize OpenCV
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }

    // Load camera calibration
    Mat cameraMatrix, distCoeffs;
    FileStorage fs("camera_parameters.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Error: Could not open camera parameters file." << endl;
        return -1;
    }
    fs["CameraMatrix"] >> cameraMatrix;
    fs["DistortionCoefficients"] >> distCoeffs;
    fs.release();

    // Initialize OpenGL
    if (!glfwInit()) {
        cout << "Error initializing GLFW" << endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(640, 480, "Augmented Reality with OpenGL", NULL, NULL);
    if (!window) {
        cout << "Error creating window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewInit();

    // Set OpenGL settings
    glEnable(GL_DEPTH_TEST);

    Size boardSize(9, 6);
    float squareSize = 1.0;

    while (!glfwWindowShouldClose(window)) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners);

        if (found) {
            vector<Point3f> objectPoints;
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));

            Mat rvec, tvec;
            solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

            // OpenGL rendering
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(60.0, 640.0 / 480.0, 0.1, 100.0);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);

            renderCube(cameraMatrix, distCoeffs, rvec, tvec);

            glfwSwapBuffers(window);
        }

        imshow("Camera Feed", frame);
        char key = waitKey(1);
        if (key == 'q') break;
    }

    cap.release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
