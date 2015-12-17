cmake_minimum_required(VERSION 2.8)
project(Ransac)

find_package(OpenCV REQUIRED)

add_executable(Ransac Ransac.cpp)

target_link_libraries(Ransac ${OpenCV_LIBS})
