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

// Constants
const float MARKER_LENGTH = 0.241f; // Marker length in meters

/**
 * @brief Displays usage information.
 * 
 * @param programName Name of the executable.
 */
void printUsage(const string& programName) {
    cerr << "Usage: " << programName << " <path_to_images> <output_yml_file> <output_corrected_images_path>" << endl;
}

/**
 * @brief Generates 3D object points for the calibration pattern.
 * 
 * @return vector<Point3f> 3D points in the real world.
 */
vector<Point3f> generateObjectPoints(float markerLength) {
    return {
        Point3f(0, 0, 0),                         // Bottom-left corner
        Point3f(markerLength, 0, 0),             // Bottom-right corner
        Point3f(markerLength, markerLength, 0),  // Top-right corner
        Point3f(0, markerLength, 0)              // Top-left corner
    };
}

/**
 * @brief Processes a single image: detects markers, refines corners, and accumulates points.
 * 
 * @param filePath Path to the image file.
 * @param detector Reference to the FractalDetector.
 * @param grayGrayGrayscale image.
 * @param image2DPoints Reference to the container for accumulated 2D image points.
 * @param realWorldPoints Reference to the container for accumulated 3D world points.
 * @return bool True if markers are detected and processed, False otherwise.
 */
bool processImage(const string& filePath, FractalDetector& detector, vector<vector<Point2f>>& image2DPoints, 
                vector<vector<Point3f>>& realWorldPoints) {
    cout << "Processing image: " << filePath << endl;

    Mat image = imread(filePath);
    if (image.empty()) {
        cerr << "Error: Unable to load image " << filePath << endl;
        return false;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Detect fractals in the image
    if (!detector.detect(gray)) {
        cout << "No fractals detected in " << filePath << endl;
        return false;
    }

    // Refine detected marker corners
    for (const auto& marker : detector.getMarkers()) {
        cout << "Detected marker ID: " << marker.id << endl;

        vector<Point2f> refinedCorners = marker;
        cornerSubPix(gray, refinedCorners, Size(11, 11), Size(-1, -1),
                    TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

        image2DPoints.emplace_back(refinedCorners);
        realWorldPoints.emplace_back(generateObjectPoints(MARKER_LENGTH));

        // Draw markers on the image
        detector.drawMarkers(image);
    }

    return true;
}

/**
 * @brief Saves the processed image to the specified directory.
 * 
 * @param originalFilePath Original image file path.
 * @param processedImage The image after processing (e.g., marker detection).
 * @param outputDir Directory where the processed image will be saved.
 * @return bool True if the image is saved successfully, False otherwise.
 */
bool saveProcessedImage(const string& originalFilePath, const Mat& processedImage, const string& outputDir) {
    // Extract the filename from the original file path
    size_t pos = originalFilePath.find_last_of("/\\");
    string filename = (pos == string::npos) ? originalFilePath : originalFilePath.substr(pos + 1);

    // Create the output file path
    string outputPath = outputDir + "/" + filename;

    // Save the image
    if (!imwrite(outputPath, processedImage)) {
        cerr << "Error: Failed to save image to " << outputPath << endl;
        return false;
    }

    cout << "Saved processed image to " << outputPath << endl;
    return true;
}

/**
 * @brief Corrects the distortion in the image using the calibration parameters and saves the corrected image.
 * 
 * @param filePath Path to the original image file.
 * @param mapX Precomputed map for x-coordinates.
 * @param mapY Precomputed map for y-coordinates.
 * @param outputDir Directory where the corrected image will be saved.
 * @return bool True if the image is saved successfully, False otherwise.
 */
bool correctAndSaveImage(const string& filePath, const Mat& mapX, const Mat& mapY, const string& outputDir) {
    Mat image = imread(filePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Unable to load image " << filePath << endl;
        return false;
    }

    Mat correctedImage;
    remap(image, correctedImage, mapX, mapY, INTER_LINEAR);

    // Save the corrected image
    return saveProcessedImage(filePath, correctedImage, outputDir);
}

int main(int argc, char** argv) {
    // Validate command-line arguments
    if (argc < 4) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    string imagesPath = argv[1];
    string outputYmlFile = argv[2];
    string correctedImagesPath = argv[3];

    // Collect image file names
    vector<String> fileNames;
    cout << "Searching for images in: " << imagesPath << endl;
    glob(imagesPath, fileNames, false);

    if (fileNames.empty()) {
        cerr << "Error: No images found in the specified path." << endl;
        return EXIT_FAILURE;
    }

    // Initialize FractalDetector
    FractalDetector detector;
    detector.setConfiguration("FRACTAL_4L_6");

    // Generate object points for the calibration pattern
    vector<Point3f> objectPoints = generateObjectPoints(MARKER_LENGTH);

    // Containers for accumulated points
    vector<vector<Point2f>> image2DPoints;
    vector<vector<Point3f>> realWorldPoints;

    // Process each image to detect markers and accumulate points
    for (const auto& file : fileNames) {
        bool success = processImage(file, detector, image2DPoints, realWorldPoints);

        if (success) {
            // Save the image with drawn markers
            Mat processedImage = imread(file);
            detector.drawMarkers(processedImage);
            if (!saveProcessedImage(file, processedImage, correctedImagesPath)) {
                cerr << "Warning: Failed to save processed image for " << file << endl;
            }
        }
    }

    if (image2DPoints.empty() || realWorldPoints.empty()) {
        cerr << "Error: Insufficient data for calibration." << endl;
        return EXIT_FAILURE;
    }

    // Initialize calibration parameters
    Matx33f intrinsicMatrix = Matx33f::eye();
    Vec<float, 5> distortionCoefficients = Vec<float, 5>::zeros();
    vector<Mat> rotationVectors, translationVectors;

    // Calibration flags
    int calibrationFlags = CALIB_FIX_K3 | CALIB_ZERO_TANGENT_DIST;

    // Determine image size from the first image
    Mat firstImage = imread(fileNames[0]);
    if (firstImage.empty()) {
        cerr << "Error: Unable to load the first image for determining frame size." << endl;
        return EXIT_FAILURE;
    }
    Size frameSize(firstImage.cols, firstImage.rows);

    cout << "Starting calibration process..." << endl;

    // Perform camera calibration
    double calibrationError = calibrateCamera(realWorldPoints, image2DPoints, frameSize,
                                            intrinsicMatrix, distortionCoefficients,
                                            rotationVectors, translationVectors,
                                            calibrationFlags);

    cout << "Calibration completed with error: " << calibrationError << endl;

    // Save calibration data to YML file
    FileStorage fs(outputYmlFile, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "Error: Unable to open file " << outputYmlFile << " for writing." << endl;
        return EXIT_FAILURE;
    }

    fs << "IntrinsicMatrix" << Mat(intrinsicMatrix);
    fs << "DistortionCoefficients" << Mat(distortionCoefficients);

    // Save Rotation Vectors
    fs << "RotationVectors" << "[";
    for (size_t i = 0; i < rotationVectors.size(); ++i) {
        fs << rotationVectors[i];
    }
    fs << "]";

    // Save Translation Vectors
    fs << "TranslationVectors" << "[";
    for (size_t i = 0; i < translationVectors.size(); ++i) {
        fs << translationVectors[i];
    }
    fs << "]";

    // Optional: Save Extrinsic Matrices
    fs << "ExtrinsicMatrices" << "[";
    for (size_t i = 0; i < rotationVectors.size(); ++i) {
        Mat R;
        Rodrigues(rotationVectors[i], R); // Convert to rotation matrix
        Mat extrinsic = Mat::eye(4, 4, R.type());
        R.copyTo(extrinsic(Range(0,3), Range(0,3)));
        translationVectors[i].copyTo(extrinsic(Range(0,3), Range(3,4)));
        fs << extrinsic;
    }
    fs << "]";

    fs.release();

    cout << "Calibration data (including extrinsic parameters) saved to " << outputYmlFile << endl;


    // Prepare undistortion maps
    Mat mapX, mapY;
    initUndistortRectifyMap(Mat(intrinsicMatrix), Mat(distortionCoefficients), Matx33f::eye(),
                            Mat(intrinsicMatrix), frameSize, CV_32FC1, mapX, mapY);

    // Correct and save all images using the calibration parameters
    cout << "Correcting and saving images..." << endl;
    for (const auto& file : fileNames) {
        if (!correctAndSaveImage(file, mapX, mapY, correctedImagesPath)) {
            cerr << "Warning: Failed to save corrected image for " << file << endl;
        }
    }

    cout << "All images have been processed and saved successfully." << endl;
    return EXIT_SUCCESS;
}

// FileStorage fs("calibration.yml", FileStorage::READ);
// if (!fs.isOpened()) {
//     cerr << "Failed to open calibration file." << endl;
//     return -1;
// }

// Mat intrinsicMatrix;
// Mat distortionCoefficients;
// vector<Mat> rotationVectors;
// vector<Mat> translationVectors;
// vector<Mat> extrinsicMatrices;

// fs["IntrinsicMatrix"] >> intrinsicMatrix;
// fs["DistortionCoefficients"] >> distortionCoefficients;
// fs["RotationVectors"] >> rotationVectors;
// fs["TranslationVectors"] >> translationVectors;
// fs["ExtrinsicMatrices"] >> extrinsicMatrices;

// fs.release();

// // Example: Print the first extrinsic matrix
// if (!extrinsicMatrices.empty()) {
//     cout << "First Extrinsic Matrix:\n" << extrinsicMatrices[0] << endl;
// }