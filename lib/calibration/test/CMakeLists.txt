# cmake .. -Daruco_DIR=/Users/tonywang/Documents/hiwi/Aries/Code/Aries/lib/calibration/aruco-3.1.12/build -DOpenCV_DIR=/Users/tonywang/Documents/hiwi/Aries/Code/Aries/lib/calibration/aruco-3.1.12/build/Release/generators  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/Users/tonywang/Documents/hiwi/Aries/Code/Aries/lib/calibration/aruco-3.1.12/build/Release/generators
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.0)
project(examples LANGUAGES CXX) 

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(OpenCV REQUIRED)
find_package(aruco REQUIRED)

message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/Users/tonywang/Documents/hiwi/Aries/Code/Aries/lib/calibration/aruco-3.1.12/src)
link_directories(/Users/tonywang/Documents/hiwi/Aries/Code/Aries/lib/calibration/aruco-3.1.12/build/src)

add_executable(sample_detection sample_detection.cpp)
target_link_libraries(sample_detection aruco ${OpenCV_LIBS})