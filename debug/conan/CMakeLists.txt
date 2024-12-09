cmake_minimum_required(VERSION 3.15)
project(opencv_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/Release/generators)

find_package(OpenCV REQUIRED CONFIG)
find_package(Eigen3 REQUIRED CONFIG)

add_executable(opencv_example main.cpp)
target_link_libraries(opencv_example PRIVATE 
    opencv::opencv
    Eigen3::Eigen
)