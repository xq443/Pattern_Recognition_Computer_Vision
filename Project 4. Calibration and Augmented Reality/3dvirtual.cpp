/*
  Xujia Qin 
  13th Mar, 2025
  S21
*/
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

using namespace cv;
using namespace std;

Size boardSize(9, 6);
float squareSize = 1.0;
Mat camera_matrix, distortion_coefficients;
GLuint textureID;

void drawHouse() {
    glBegin(GL_QUADS);
    glColor3f(0.2f, 0.8f, 0.3f);
    glVertex3f(1, 1, 0);
    glVertex3f(4, 1, 0);
    glVertex3f(4, 4, 0);
    glVertex3f(1, 4, 0);
    
    glColor3f(0.6f, 0.2f, 0.2f);
    glVertex3f(1, 1, 3);
    glVertex3f(4, 1, 3);
    glVertex3f(4, 4, 3);
    glVertex3f(1, 4, 3);
    glEnd();
    
    glBegin(GL_TRIANGLES);
    glColor3f(0.9f, 0.2f, 0.2f);
    glVertex3f(1, 1, 3);
    glVertex3f(4, 1, 3);
    glVertex3f(2.5, 0.5, 5);
    glVertex3f(1, 4, 3);
    glVertex3f(4, 4, 3);
    glVertex3f(2.5, 4.5, 5);
    glEnd();
}

void drawBackground(Mat &frame) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
    
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, 1);
    glTexCoord2f(0, 0); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error: Could not open camera." << endl;
        return -1;
    }
    
    FileStorage fs("camera_parameters.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Error: Could not open camera parameters file." << endl;
        return -1;
    }
    fs["CameraMatrix"] >> camera_matrix;
    fs["DistortionCoefficients"] >> distortion_coefficients;
    fs.release();
    
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW!" << endl;
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Augmented Reality", NULL, NULL);
    if (!window) {
        cerr << "Failed to create GLFW window!" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewInit();
    
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    while (!glfwWindowShouldClose(window)) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawBackground(frame);
        
        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            vector<Point3f> objectPoints;
            for (int i = 0; i < boardSize.height; i++) {
                for (int j = 0; j < boardSize.width; j++) {
                    objectPoints.push_back(Point3f(j * squareSize, i * squareSize, 0));
                }
            }
            Mat rvec, tvec;
            solvePnP(objectPoints, corners, camera_matrix, distortion_coefficients, rvec, tvec);
            
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(45.0, 800.0 / 600.0, 0.1, 100.0);
            
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0);
            
            glTranslatef(tvec.at<double>(0), tvec.at<double>(1), -tvec.at<double>(2));
            drawHouse();
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    cap.release();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
