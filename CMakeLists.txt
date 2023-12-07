cmake_minimum_required(VERSION 3.0.0)
project(Med_TA)

set(CMAKE_CXX_STANDARD 17)

add_executable(Med_TA yolov5.cpp
        jsoncpp.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Med_TA ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
