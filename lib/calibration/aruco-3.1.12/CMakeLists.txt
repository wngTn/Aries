cmake_minimum_required(VERSION 3.10)

# Project Name and Description
project(ArucoCalibration
        VERSION 1.0
        DESCRIPTION "Aruco Marker Calibration Tool"
        LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optionally, set the build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable Debug Symbols if in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found. Please install OpenCV.")
endif()

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found. Please install Eigen3.")
endif()

# Find OpenGL
find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message(FATAL_ERROR "OpenGL not found")
endif()

# Find GLUT
find_package(GLUT REQUIRED)
if(NOT GLUT_FOUND)
    message(FATAL_ERROR "GLUT not found")
endif()

# Include Directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/3rdparty/eigen3
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/utils_calibration
    ${CMAKE_SOURCE_DIR}/utils_dcf
    ${CMAKE_SOURCE_DIR}/utils_fractal
    ${CMAKE_SOURCE_DIR}/utils_markermap
    ${CMAKE_SOURCE_DIR}/utils_gl
    ${OPENGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
)

# Collect Source Files
# Manually listing source files for better control

# Source files from src/
set(SRC_FILES
    # src/aruco.cpp
    src/cameraparameters.cpp
    src/cvdrawingutils.cpp
    src/debug.cpp
    src/dictionary.cpp
    src/dictionary_based.cpp
    src/fractaldetector.cpp
    src/ippe.cpp
    src/marker.cpp
    src/markerdetector.cpp
    src/markerdetector_impl.cpp
    src/markerlabeler.cpp
    src/markermap.cpp
    src/posetracker.cpp
    # src/timers.cpp  # Ensure this file exists
)

# Source files from utils/
set(UTILS_FILES
    utils/aruco_batch_processing.cpp
    utils/aruco_dummy.cpp
    utils/aruco_print_dictionary.cpp
    utils/aruco_print_marker.cpp
    utils/aruco_simple.cpp
    utils/aruco_test.cpp
    utils/aruco_tracker.cpp
    utils/opencv_calibration.cpp
)

# Source files from utils_calibration/
set(UTILS_CALIBRATION_FILES
    utils_calibration/aruco_calibration.cpp
    utils_calibration/aruco_calibration_fromimages.cpp
    utils_calibration/aruco_portable_calibration.cpp
    utils_calibration/calibrator.cpp
)

# Source files from utils_dcf/
set(UTILS_DCF_FILES
    utils_dcf/aruco_dcf.cpp
    utils_dcf/aruco_dcf_markermap.cpp
    utils_dcf/pcdwriter.cpp
)

# Source files from utils_fractal/
set(UTILS_FRACTAL_FILES
    utils_fractal/fractal_create.cpp
    utils_fractal/fractal_pix2meters.cpp
    utils_fractal/fractal_print_marker.cpp
    utils_fractal/fractal_tracker.cpp
)

# Source files from utils_markermap/
set(UTILS_MARKERMAP_FILES
    utils_markermap/aruco_create_markermap.cpp
    utils_markermap/aruco_markermap_pix2meters.cpp
    utils_markermap/aruco_simple_markermap.cpp
    utils_markermap/aruco_test_markermap.cpp
    utils_markermap/pcdwriter.cpp
)

# Source files from utils_gl/
set(UTILS_GL_FILES
    utils_gl/aruco_test_board_gl_mask.cpp
    utils_gl/aruco_test_gl.cpp
    utils_gl/aruco_test_markermap_gl.cpp
)

# Combine all source files
set(ALL_SOURCE_FILES
    ${SRC_FILES}
    ${UTILS_FILES}
    ${UTILS_CALIBRATION_FILES}
    ${UTILS_DCF_FILES}
    ${UTILS_FRACTAL_FILES}
    ${UTILS_MARKERMAP_FILES}
    ${UTILS_GL_FILES}
    main.cpp
)

# Add Executable
add_executable(aruco_calibration ${ALL_SOURCE_FILES})

# Link Libraries
target_link_libraries(aruco_calibration
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
    # Add other necessary libraries here, for example:
    # pthread
)

# Define preprocessor macro to silence OpenGL deprecation warnings on macOS
target_compile_definitions(aruco_calibration PRIVATE GL_SILENCE_DEPRECATION)

# Installation (Optional)
install(TARGETS aruco_calibration
        RUNTIME DESTINATION bin)

# Enable CPack if you want to create packages
# include(CPack)


# # cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$(pwd)/Release/generators
# # ----------------------------------------------------------------------------
# #   Basic Configuration
# # ----------------------------------------------------------------------------
# cmake_minimum_required(VERSION 3.0)
# project(aruco VERSION "3.1.12" LANGUAGES CXX)
# set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
# set(CMAKE_INCLUDE_CURRENT_DIR ON)
# set(CMAKE_CXX_STANDARD 11) # C++11...
# set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
# set(CMAKE_CXX_EXTENSIONS ON) #...with compiler extensions like gnu++11

# include(cmake/options.cmake)

# include(cmake/findDependencies.cmake)
# include(cmake/compilerOptions.cmake)


# add_subdirectory(src)

# if(GL_FOUND AND BUILD_GLSAMPLES)
#   add_subdirectory(utils_gl)
# endif()
# if(BUILD_TESTS)
#   add_subdirectory(tests)
# endif()


# if(BUILD_UTILS)
#   add_subdirectory(utils)
#   add_subdirectory(utils_markermap)
#   add_subdirectory(utils_calibration)
#   add_subdirectory(utils_dcf)
#   IF(BUILD_SVM)
#   ADD_SUBDIRECTORY(utils_svm)
#   ENDIF()
#    ADD_SUBDIRECTORY(utils_fractal)
# endif()
# include(cmake/printInfo.cmake)
# include(cmake/installOptions.cmake)

# # Package Generator  #######################################################
# IF(BUILD_DEBPACKAGE)
# include(cmake/cpack.cmake)
# ENDIF()
