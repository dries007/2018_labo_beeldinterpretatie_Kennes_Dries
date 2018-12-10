cmake_minimum_required(VERSION 3.12)
project(labo_opencv)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(labo_opencv_faces faces.cpp)
target_link_libraries(labo_opencv_faces ${OpenCV_LIBS})

add_executable(labo_opencv_people people.cpp)
target_link_libraries(labo_opencv_people ${OpenCV_LIBS})
