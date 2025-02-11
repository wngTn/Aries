INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)
find_package(nlohmann_json REQUIRED)  # Add this line

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${nlohmann_json_INCLUDE_DIRS})  # Add this line

if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
SET(THREADLIB "pthread")
ENDIF()
add_executable(aruco_calibration aruco_calibration.cpp calibrator.h calibrator.cpp )
add_executable(aruco_calibration_fromimages aruco_calibration_fromimages.cpp dirreader.h)
add_executable(main_aruco fractal_calibration.cpp dirreader.h)

# Replace the add_executable with pybind11_add_module for the Python module
pybind11_add_module(calibration_module calibration_module.cpp)

target_link_libraries(aruco_calibration             aruco opencv_calib3d opencv_highgui ${THREADLIB})
target_link_libraries(aruco_calibration_fromimages  aruco opencv_calib3d opencv_highgui ${THREADLIB})
target_link_libraries(main_aruco  aruco opencv_calib3d opencv_highgui ${THREADLIB})

# Link libraries to your Python module
target_link_libraries(calibration_module PRIVATE 
    aruco
    ${OpenCV_LIBS}
    ${THREADLIB}
    nlohmann_json::nlohmann_json  # Add if you're using nlohmann::json
)

install(TARGETS      aruco_calibration aruco_calibration_fromimages main_aruco RUNTIME DESTINATION bin)

IF(WIN32)
install(FILES "${PROJECT_SOURCE_DIR}/utils_calibration/aruco_calibration_grid_board_a4.pdf"  DESTINATION bin/)
#install(FILES "${PROJECT_SOURCE_DIR}/utils/myown.dict"   DESTINATION bin/)
elseif(UNIX)
    install(FILES "aruco_calibration_grid_board_a4.pdf"  DESTINATION share/${PROJECT_NAME})
   #install(FILES "${PROJECT_SOURCE_DIR}/utils/myown.dict"   DESTINATION  shared/${PROJECT_NAME})
ENDIF()

