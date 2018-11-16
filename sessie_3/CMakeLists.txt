cmake_minimum_required(VERSION 3.12)
project(labo_opencv)

set(CMAKE_CXX_STANDARD 98)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(labo_opencv main.cpp)
target_link_libraries(labo_opencv ${OpenCV_LIBS})
