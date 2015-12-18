#include "ransaccer.h"
#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
    string path_easy{"../src/marker_corny/marker_corny_"};
    string path_hard{"../src/marker_corny_hard/marker_corny_hard_"};
    string save_path{"../results/tracked_marker_corny_"};
    string type{".png"};
    string save_type{".jpg"};
    string file{""};
    string save_file{""};
    Point2f p1;
    auto t_start = std::chrono::system_clock::now();
    Ransac rs;
    Mat to_write;
    for (int i{1}; i < 31; i++) {
        if (i < 10) {
            file = path_easy + "0" + to_string(i) + type;
            save_file = save_path + "0" + to_string(i) + save_type;
        } else {
            file = path_easy + to_string(i) + type;
            save_file = save_path + to_string(i) + save_type;
        }
        rs.assign(file);
        p1 = rs.extract();
        to_write = rs.draw_stuff();
        imwrite(save_file, to_write);
    }
    auto t_end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start);
    double fps = 30. / duration.count();
    save_path += "hard_";
    cout << "FPS, easy: " << fps << endl;
    t_start = std::chrono::system_clock::now();
    for (int i{1}; i < 53; i++) {
        if (i < 10) {
            file = path_hard + "0" + to_string(i) + type;
            save_file = save_path + "0" + to_string(i) + save_type;
        } else {
            file = path_hard + to_string(i) + type;
            save_file = save_path + to_string(i) + save_type;
        }
        rs.assign(file);
        p1 = rs.extract();
        to_write = rs.draw_stuff();
        imwrite(save_file, to_write);
    }
    t_end = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start);
    fps = 30. / duration.count();
    cout << "FPS, hard: " << fps << endl;


    return 0;
}
