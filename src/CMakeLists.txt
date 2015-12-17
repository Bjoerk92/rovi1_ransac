cmake_minimum_required(VERSION 2.8)
project(Ransac)

find_package(OpenCV REQUIRED)

add_executable(Ransac ransaccer_driver.cpp ransaccer.cpp)

target_link_libraries(Ransac ${OpenCV_LIBS})
