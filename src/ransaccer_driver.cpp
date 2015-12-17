#include "ransaccer.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
    Point center;
    Mat image = imread("../src/marker_corny/marker_corny_01.png", CV_LOAD_IMAGE_COLOR);
    Ransac rs;
    rs.assign(image);
    center = rs.extract();
    cout << center << endl;
    return 0;
}
