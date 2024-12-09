#include <algorithm>


#include <fstream>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "aruco.h"
#include "aruco_calibration_grid_board_a4.h"
#include "dirreader.h"
#include <stdexcept>

using namespace std;
using namespace cv;
using namespace aruco;

int main(int argc, char **argv) {

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <path_to_images> <name_yml_file>" << endl;
        return -1;
    }

    // Path of the images (ArUco markers)
    vector<String> fileNames;
    cout << "Searching images in: " << argv[1] << endl;
    glob(argv[1], fileNames, false);  // Change the image path

    // Defining the marker size (length in meters)
    const float markerLength = 0.241f;  // Length of the marker in meters (adjust as needed)

    // Defining the ArUco marker dictionary
    
    FractalDetector FDetector;
    FDetector.setConfiguration("FRACTAL_4L_6");

    ///////////////////////////////////////////////////////////////////////////
    // 1. Dectecting feature 

    // 3D points for the marker (assuming all markers are in the XY plane)
    vector<Point3f> objPoints = {
        Point3f(0, 0, 0),  // Bottom-left corner
        Point3f(markerLength, 0, 0),  // Bottom-right corner
        Point3f(markerLength, markerLength, 0),  // Top-right corner
        Point3f(0, markerLength, 0)  // Top-left corner
    };

    // Stores the 2D detected
    vector<vector<Point2f>> image2Dpoints;
    
    // Stores the 3D points in the real world
    vector<vector<Point3f>> realWorldPoints;

    // We use these points to make a comparison between them in order to define
    // how the camera proyects the 3D points of the real world in the 2D image
    // so we can obtain later the calibration parameters

    // Loop through each image
    for (auto const &file : fileNames) {

        /////////////////////////////////////////////////////////////
        // 1.1 Process the image in order to detect the pattern
        cout << "Processing image: " << string(file) << endl;

        Mat img = imread(file);
        Mat gray;

        // Convert the image to grayscale
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // Detect ArUco markers in the image
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners, rejectedCandidates;

        // Detect fractals in the image using FDetector
        if (!FDetector.detect(gray)) {
            cout << "No fractals detected in " << string(file) << endl;
            continue;  // Skip to the next image
        }

        /////////////////////////////////////////////////////////////
        // 1.2 Refining the corners detected to get accuracy

        for (auto m : FDetector.getMarkers()) {
            cout << "Detected marker ID: " << m.id << endl;

            // Refine corners for accuracy
            vector<Point2f> corners = m;  // Assume `m` provides access to corner points
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                        TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

            // Add the 2D points to the list
            image2Dpoints.push_back(corners);

            // Add the corresponding 3D points
            realWorldPoints.push_back(objPoints);
        }

        
        /////////////////////////////////////////////////////////////
        // 1.3 Display

        FDetector.drawMarkers(img);

        Mat image_resized; // no need to use this, I used it to adapt the program to my screen
        resize(img, image_resized, Size(1500, 1000)); // you can choose your preferred one

        imshow("ArUco Marker Detection", image_resized);
        waitKey(0);  // Wait until a key is pressed

        
    }

    ///////////////////////////////////////////////////////////////////////////
    // 2. Getting Intrinsic Matrix and Distortion coefficients

    Matx33f intrinsicMatrix(Matx33f::eye());
    Vec<float, 5> distortionCoef(0, 0, 0, 0, 0);
    vector<Mat> rotationVect, translationVect;

    //int flags = CALIB_FIX_ASPECT_RATIO + CALIB_FIX_K3 + CALIB_ZERO_TANGENT_DIST + CALIB_FIX_PRINCIPAL_POINT;
    int flags = CALIB_FIX_K3 | CALIB_ZERO_TANGENT_DIST;
    //int flags = 0;
    
    Mat img = imread(fileNames[0]);  // Read the first image to detect the size
    Size frameSize(img.cols, img.rows);


    cout << "Starting calibration process..." << endl;

    /////////////////////////////////////////////////////////////
    // 2.1 Error, intrinsic matrix and distorsion coefficients

    float error = calibrateCamera(realWorldPoints, image2Dpoints, frameSize,
                                  intrinsicMatrix, distortionCoef, rotationVect,
                                  translationVect, flags);

    cout << endl << "Calibration error: " << error << endl;
    //cout << "Intrinsic Matrix: " << endl << intrinsicMatrix << endl << endl;
    //cout << "Distortion Coefficients: " << endl << distortionCoef << endl << endl;

    ///////////////////////////////////////////////////////////////////////////
    // 3. Correcting images to test the calibration accuracy

    /////////////////////////////////////////////////////////////
    // 3.1 Getting the fixed position of every pixel

    Mat mapX, mapY;
    initUndistortRectifyMap(intrinsicMatrix, distortionCoef, Matx33f::eye(),
                            intrinsicMatrix, frameSize, CV_32FC1, mapX, mapY);

    /*
    * mapX and mapY are the matrix that contain the fixed position of every
    * pixel of the image to correct the distortion
    * 
    * e.g. if the value of mapX[100, 50] = 98 and mapY[100, 50] = 49, 
    * the pixel in (100, 50) of the original image should be map into 
    * the position (98, 49)
    */

   

    /////////////////////////////////////////////////////////////
    // 3.2 Show corrected images

    for (auto const &file: fileNames) {

        cout << string(file) << endl;

        Mat img = imread(file, IMREAD_COLOR);
        Mat imgCorrected;

        // applying the transformation matrixes obtained previously
        remap(img, imgCorrected, mapX, mapY, INTER_LINEAR);

        Mat image_resized; // no need to use this, I used it to adapt the program to my screen
        resize(imgCorrected, image_resized, Size(1500, 1000)); // you can choose your preferred one

        imshow("Corrected image ", image_resized);
        waitKey(0);
    }

    

    ///////////////////////////////////////////////////////////////////////////
    // 4. Exporting calibration data to a YML file

    cout << endl << "Exporting calibration data to " << argv[2] << endl;

    // CameraParameters 
    CameraParameters camp; 
    camp.setParams(Mat(intrinsicMatrix), Mat(distortionCoef), frameSize);

    camp.saveToFile(argv[2]); 
    cout << "File saved in succesfully" << endl; 
    
    /*
    No need to show this, just if you want
    
    // Rotation vector
    cout << "TYPE=" << (rotationVect[0].type() == CV_64F) << endl;
    cout << "vector<Mat> rotationVect;" << endl; 
    for (size_t i = 0; i < rotationVect.size(); i++) {
        double *ptr = rotationVect[i].ptr<double>(0); 
        cout << "rotationVect.push_back( (Mat_<double>(3,1) << " << ptr[0] << "," << ptr[1] << "," << ptr[2] << "));" << endl; 
    } 
    
    // Translation Vector
    cout << "vector<Mat> translationVect;" << endl; 
    for (size_t i = 0; i < translationVect.size(); i++) { 
        double *ptr = translationVect[i].ptr<double>(0); 
        cout << "translationVect.push_back( (Mat_<double>(3,1) << " << ptr[0] << "," << ptr[1] << "," << ptr[2] << "));" << endl;
    }
    */

    return 0;
}
