/*
Compile of the module to execute it in pyhton (python 3.12):
g++ -shared -fPIC -I../src -I/usr/include/opencv4 -I/usr/include/opencv -I/path/to/nlohmann/json -I/usr/include/python3.12 -L/usr/lib -L/usr/local/lib src/utils/calibration_module.cpp -o src/utils/calibration_module.so -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_imgcodecs -laruco -lpython3.12

Usage:
python3.12 src/run/calibrate_marshall.py data/trial_phone_camera/marshall/ data/trial_phone_camera/phoneCamera.json 0.173

Note:
marker_length = 0.173f in the trial case
*/

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
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;
using namespace cv;
using namespace aruco;
using json = nlohmann::json;

std::vector<cv::Point2f> detect_marker(const std::string& path_to_image)
{
    // Initialize return vector
    std::vector<cv::Point2f> image2Dpoints;
    
    // Defining the ArUco marker dictionary
    FractalDetector FDetector;
    FDetector.setConfiguration("FRACTAL_4L_6");
    Mat img = imread(path_to_image);
    
    // Check if the image was loaded successfully
    if (img.empty())
    {
        cerr << "Error loading image: " << path_to_image << endl;
        throw runtime_error("Error loading image");
    }
    
    Mat gray;
    // Convert the image to grayscale
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    // Detect fractals in the image using FDetector
    if (!FDetector.detect(gray))
    {
        cout << "No fractals detected in " << path_to_image << endl;
        return image2Dpoints; // Return empty vector if no fractals detected
    }
    
    for (const auto& m : FDetector.getMarkers())
    {
        if (m.id != 0) {
            break;
        }
        cout << "Detected marker ID: " << m.id << endl;
        
        // Refine corners for accuracy
        vector<Point2f> corners = m; // Assume `m` provides access to corner points
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                    TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));
        
        // Add each corner point individually to image2Dpoints
        for (const auto& corner : corners) {
            image2Dpoints.push_back(corner);
        }
    }
    
    return image2Dpoints;
}

void calibrate_camera(String path_to_images, String name_json_file, int camera_num, int num_images = -1)
{
    const float markerLength = 0.54f; // Fixed marker length

    // Path of the images (ArUco markers)
    vector<String> fileNames;

    // Create the glob pattern for specific camera images
    String pattern = path_to_images + "*_camera" + format("%02d", camera_num) + ".jpg";
    cout << "Searching images in: " << pattern << endl;

    glob(pattern, fileNames, false);

    if (fileNames.empty())
    {
        cerr << "No images found for camera " << camera_num << endl;
        return;
    }

    // Handle image sampling
    vector<String> sampledFiles;
    if (num_images == -1 || num_images >= fileNames.size())
    {
        sampledFiles = fileNames;
    }
    else
    {
        // Ensure we don't try to sample more images than available
        num_images = min(num_images, static_cast<int>(fileNames.size()));

        // Calculate sampling interval
        double step = static_cast<double>(fileNames.size()) / num_images;

        // Perform uniform sampling
        for (int i = 0; i < num_images; i++)
        {
            int index = static_cast<int>(i * step);
            sampledFiles.push_back(fileNames[index]);
        }
    }

    cout << "Using " << sampledFiles.size() << " out of " << fileNames.size() << " available images for camera " << camera_num << endl;

    // Defining the ArUco marker dictionary
    FractalDetector FDetector;
    FDetector.setConfiguration("FRACTAL_4L_6");

    ///////////////////////////////////////////////////////////////////////////
    // 1. Dectecting feature

    // 3D points for the marker (assuming all markers are in the XY plane)
    vector<Point3f> objPoints = {
        Point3f(0, 0, 0),                       // Bottom-left corner
        Point3f(markerLength, 0, 0),            // Bottom-right corner
        Point3f(markerLength, markerLength, 0), // Top-right corner
        Point3f(0, markerLength, 0)             // Top-left corner
    };
    //     Point3f(markerLength/2, 0, -markerLength/2),            // Bottom-left corner
    //     Point3f(-markerLength/2, 0, -markerLength/2),           // Bottom-right corner
    //     Point3f(-markerLength/2, 0, markerLength/2),            // Top-right corner
    //     Point3f(markerLength/2, 0, markerLength/2),             // Top-left corner
    // };

    // Stores the 2D detected
    vector<vector<Point2f>> image2Dpoints;

    // Stores the 3D points in the real world
    vector<vector<Point3f>> realWorldPoints;

    // Loop through each sampled image
    for (const auto &file : sampledFiles)
    {
        cout << "Processing image: " << string(file) << endl;

        Mat img = imread(file);
        Mat gray;

        // Convert the image to grayscale
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // Detect fractals in the image using FDetector
        if (!FDetector.detect(gray))
        {
            cout << "No fractals detected in " << string(file) << endl;
            continue; // Skip to the next image
        }

        /////////////////////////////////////////////////////////////
        // 1.2 Refining the corners detected to get accuracy

        for (const auto &m : FDetector.getMarkers())
        {
            if (m.id != 0)
            {
                break;
            }

            cout << "Detected marker ID: " << m.id << endl;

            // Refine corners for accuracy
            vector<Point2f> corners = m; // Assume `m` provides access to corner points
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

            // Log corner points
            for (const auto &c : corners)
            {
                cout << "Corner: " << c << endl;
            }

            // Add the 2D points to the list
            image2Dpoints.push_back(corners);

            // Add the corresponding 3D points
            realWorldPoints.push_back(objPoints);
        }

        /////////////////////////////////////////////////////////////
        // 1.3 Display

        FDetector.drawMarkers(img);

        imshow("ArUco Marker Detection", img);
        waitKey(0);
    }

    Mat img = imread(sampledFiles[0]); // Read the first image to detect the size
    if (img.empty())
    {
        cerr << "Error loading image: " << sampledFiles[0] << endl;
        throw runtime_error("Error loading image");
    }

    ///////////////////////////////////////////////////////////////////////////
    // 2. Getting Intrinsic Matrix and Distortion coefficients

    Matx33f intrinsicMatrix(Matx33f::eye()); // Initialize intrinsic matrix with reasonable starting values
    Vec<float, 5> distortionCoef(0, 0, 0, 0, 0);
    vector<Mat> rotationVect, translationVect;

    // int flags = CALIB_FIX_ASPECT_RATIO + CALIB_FIX_K3 + CALIB_ZERO_TANGENT_DIST + CALIB_FIX_PRINCIPAL_POINT;
    // int flags = CALIB_FIX_K3 | CALIB_ZERO_TANGENT_DIST;
    // int flags = CALIB_USE_INTRINSIC_GUESS;
    int flags = 0;

    if (img.empty())
    {
        cerr << "Error loading image: " << fileNames[0] << endl;
        throw runtime_error("Error loading image");
    }

    Size frameSize(img.cols, img.rows);

    cout << endl
         << "Starting calibration process..." << endl;

    /////////////////////////////////////////////////////////////
    // 2.1 Calibration error, intrinsic matrix and distorsion coefficients

    float error = calibrateCamera(
        realWorldPoints,
        image2Dpoints,
        frameSize,
        intrinsicMatrix,
        distortionCoef,
        rotationVect,
        translationVect,
        flags);

    cout << "Calibration error: " << error << endl
         << endl;

    ///////////////////////////////////////////////////////////////////////////
    // 5. Correcting images to test the calibration accuracy

    /////////////////////////////////////////////////////////////
    // 5.1 Getting the fixed position of every pixel

    // Correcting images to test the calibration accuracy
    Mat mapX, mapY;
    initUndistortRectifyMap(
        intrinsicMatrix,
        Mat(distortionCoef),
        Matx33f::eye(),
        intrinsicMatrix,
        frameSize,
        CV_32FC1,
        mapX,
        mapY);

    /////////////////////////////////////////////////////////////
    // 5.2 Show corrected images

    for (const auto &file : sampledFiles)
    {
        cout << string(file) << endl;

        Mat img = imread(file, IMREAD_COLOR);
        Mat imgCorrected;

        remap(img, imgCorrected, mapX, mapY, INTER_LINEAR);

        // Mat image_resized;
        // resize(imgCorrected, image_resized, Size(1500, 1000));

        imshow("Corrected image ", imgCorrected);
        waitKey(0);
    }

    /////////////////////////////////////////////////////////////
    // 2.2 Metric radius
    double metric_radius = sqrt(
        pow(frameSize.width / (2.0 * intrinsicMatrix(0, 0)), 2) +
        pow(frameSize.height / (2.0 * intrinsicMatrix(1, 1)), 2));

    ///////////////////////////////////////////////////////////////////////////
    // 3. Exporting calibration data to a JSON file

    cout << endl
         << "Exporting calibration data to " << name_json_file << endl;

    json calibration_data;

    calibration_data["value0"]["color_resolution"] = 6; // HD mode
    calibration_data["value0"]["color_parameters"] = {
        {"fov_x", intrinsicMatrix(0, 0)},
        {"fov_y", intrinsicMatrix(1, 1)},
        {"c_x", intrinsicMatrix(0, 2)},
        {"c_y", intrinsicMatrix(1, 2)},
        {"width", frameSize.width},
        {"height", frameSize.height},
        {"intrinsics_matrix", {{"m00", intrinsicMatrix(0, 0)}, {"m10", 0.0}, {"m20", intrinsicMatrix(0, 2)}, {"m01", 0.0}, {"m11", intrinsicMatrix(1, 1)}, {"m21", intrinsicMatrix(1, 2)}, {"m02", 0.0}, {"m12", 0.0}, {"m22", 1.0}}},
        {"radial_distortion", {
                                  {"m00", distortionCoef[0]}, {"m10", distortionCoef[1]}, {"m20", distortionCoef[4]}, {"m30", 0.0}, {"m40", 0.0}, {"m50", 0.0} // Extend as necessary for additional coefficients
                              }},
        {"tangential_distortion", {{"m00", distortionCoef[2]}, {"m10", distortionCoef[3]}}},
        {"metric_radius", metric_radius}};

    ///////////////////////////////////////////////////////////////////////////
    // 4. Calculating and exporting camera pose to the JSON file

    // Get the first transformation matrices
    // Mat rotationVect_first = rotationVect[0];
    // Mat translationVect_first = translationVect[0];

    Mat rotationVect_first, translationVect_first;

    // 3D points for the marker (assuming all markers are in the XY plane)
    vector<Point3f> markerPoints = {
        Point3f(-markerLength / 2, -markerLength / 2, 0), // Bottom-left corner
        Point3f(markerLength / 2, -markerLength / 2, 0),  // Bottom-right corner
        Point3f(markerLength / 2, markerLength / 2, 0),   // Top-right corner
        Point3f(-markerLength / 2, markerLength / 2, 0),  // Top-left corner
    };
    solvePnP(markerPoints, image2Dpoints[0], intrinsicMatrix, distortionCoef, rotationVect_first, translationVect_first);

    // Convertion to CV_64F to get accuracy, because often openCV returns CV_32F
    if (translationVect_first.type() != CV_64F)
    {
        translationVect_first.convertTo(translationVect_first, CV_64F);
    }

    if (translationVect_first.rows != 3 || translationVect_first.cols != 1)
    {
        cerr << "Error: Translation vector has incorrect dimensions." << endl;
        throw runtime_error("Error: Translation vector has incorrect dimensions.");
    }

    // Converting rotation vector to rotation matrix 3x3
    Mat rotationMatrix;
    Rodrigues(rotationVect_first, rotationMatrix);

    // Converting rotation matrix to quaternion
    Vec4f quaternion;
    Matx33f rotMat;
    rotationMatrix.copyTo(rotMat);

    /*
    The quaternion is a representation of 3D rotation consisting of four components:
        w: Scalar component (related to the angle of rotation)
        x, y, z: Vector components (rotation axis)
    */

    quaternion[0] = rotMat(2, 1) - rotMat(1, 2);
    quaternion[1] = rotMat(0, 2) - rotMat(2, 0);
    quaternion[2] = rotMat(1, 0) - rotMat(0, 1);
    quaternion[3] = rotMat(0, 0) + rotMat(1, 1) + rotMat(2, 2);

    // Normalization of the quaternion
    float norm = sqrt(quaternion[0] * quaternion[0] +
                      quaternion[1] * quaternion[1] +
                      quaternion[2] * quaternion[2] +
                      quaternion[3] * quaternion[3]);
    quaternion[0] /= norm;
    quaternion[1] /= norm;
    quaternion[2] /= norm;
    quaternion[3] /= norm;

    // Sub-object for translation
    json translation_json = {
        {"m00", translationVect_first.at<double>(0)},
        {"m10", translationVect_first.at<double>(1)},
        {"m20", translationVect_first.at<double>(2)}};

    // Sub-object for rotation
    json rotation_json = {
        {"x", quaternion[0]},
        {"y", quaternion[1]},
        {"z", quaternion[2]},
        {"w", quaternion[3]}};

    calibration_data["value0"]["camera_pose"] = {
        {"translation", translation_json},
        {"rotation", rotation_json}};

    calibration_data["value0"]["is_valid"] = true;

    ofstream json_file(name_json_file);
    json_file << calibration_data.dump(4);
    json_file.close();

    cout << "File saved successfully" << endl;
}

// Exporting the function to Python
PYBIND11_MODULE(calibration_module, m)
{
    m.doc() = "Camera calibration and marker detection using ArUco markers";

    // Convert OpenCV types to/from Python
    pybind11::class_<cv::Point2f>(m, "Point2f")
        .def(pybind11::init<float, float>())
        .def_readwrite("x", &cv::Point2f::x)
        .def_readwrite("y", &cv::Point2f::y);

    // Add both functions to the same module
    m.def("calibrate_camera", 
          &calibrate_camera, 
          "Calibrate the camera", 
          pybind11::arg("path_to_images"), 
          pybind11::arg("output_json"), 
          pybind11::arg("camera_num"), 
          pybind11::arg("num_images") = -1);

    m.def("detect_marker",
          &detect_marker,
          "Detect ArUco markers in an image",
          pybind11::arg("path_to_image"),
          pybind11::return_value_policy::move);

    // Register exception translation
    pybind11::register_exception<std::runtime_error>(m, "RuntimeError");
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " <path_to_images> <output_json> <marker_length>" << endl;
        return 1;
    }

    String path_to_images = argv[1];
    String name_json_file = argv[2];
    float marker_length = atof(argv[3]);

    calibrate_camera(path_to_images, name_json_file, marker_length);

    return 0;
}