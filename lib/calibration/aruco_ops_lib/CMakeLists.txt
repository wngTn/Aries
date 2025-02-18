cmake_minimum_required(VERSION 3.10)
project(aruco_ops)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 REQUIRED)  # Added Eigen requirement

# Get all source files
file(GLOB_RECURSE ARUCO_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/_ext-src/include/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/_ext-src/src/*.cpp"
)

pybind11_add_module(aruco_ops ${ARUCO_SOURCES})

target_include_directories(aruco_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/_ext-src/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}  # Added Eigen include dirs
)

target_link_libraries(aruco_ops PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen  # Added Eigen linkage
)