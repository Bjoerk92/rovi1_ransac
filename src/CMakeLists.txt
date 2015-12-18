cmake_minimum_required(VERSION 2.8)
project(Ransac)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
# add_definitions(-DDEBUG)

add_executable(Ransac main.cpp ransaccer.cpp)

target_link_libraries(Ransac ${OpenCV_LIBS})
