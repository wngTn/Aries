#include <iostream>
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>
#include "aruco.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<cv::Point2f> detect_marker(const std::string& path_to_image)
{
    // Initialize return vector
    std::vector<cv::Point2f> image2Dpoints;
    
    // Defining the ArUco marker dictionary
    aruco::FractalDetector FDetector;
    FDetector.setConfiguration("FRACTAL_4L_6");
    cv::Mat img = cv::imread(path_to_image);
    
    // Check if the image was loaded successfully
    if (img.empty())
    {
        std::cerr << "Error loading image: " << path_to_image << std::endl;
        throw std::runtime_error("Error loading image");
    }
    
    cv::Mat gray;
    // Convert the image to grayscale
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // Detect fractals in the image using FDetector
    if (!FDetector.detect(gray))
    {
        std::cout << "No fractals detected in " << path_to_image << std::endl;
        return image2Dpoints; // Return empty vector if no fractals detected
    }
    
    for (const auto& m : FDetector.getMarkers())
    {
        if (m.id != 0) {
            break;
        }
        std::cout << "Detected marker ID: " << m.id << std::endl;
        
        // Refine corners for accuracy
        std::vector<cv::Point2f> corners = m; // Assume `m` provides access to corner points
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
        
        // Add each corner point individually to image2Dpoints
        for (const auto& corner : corners) {
            image2Dpoints.push_back(corner);
        }
    }
    
    return image2Dpoints;
}

// Exporting the function to Python
PYBIND11_MODULE(aruco_ops, m)
{
    m.doc() = "Camera calibration and marker detection using ArUco markers";
    // Convert OpenCV types to/from Python
    pybind11::class_<cv::Point2f>(m, "Point2f")
        .def(pybind11::init<float, float>())
        .def_readwrite("x", &cv::Point2f::x)
        .def_readwrite("y", &cv::Point2f::y);

    m.def("detect_marker",
          &detect_marker,
          "Detect ArUco markers in an image",
          pybind11::arg("path_to_image"),
          pybind11::return_value_policy::move);

    // Register exception translation
    pybind11::register_exception<std::runtime_error>(m, "RuntimeError");
}