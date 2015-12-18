#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "circle_extractor.h"
#include "circ_extr_help.h"

using namespace std;
using namespace cv;

int main() { // driver for circle_extractor
	circle_extractor ce;
	Point2f p1, p2, p3, p0;
	string path{"../imgs/marker_color_"};
	string type{".png"};
	string file{""};
	// auto t_start = chrono::high_resolution_clock::now();
	Mat to_write;
	for (int i{1}; i < 30; i++) {
		if (i < 10) {
			file = path + "0" + to_string(i) + type;
		} else {
			file = path + to_string(i) + type;
		}
		ce.assign(file);
		p0 = ce.extract();
		ce.extract(p1, p2, p3);
		cout << p1 << p2 << p3 << endl;
	}

	path = path + "hard_";
	for (int i{1}; i < 53; i++) {
		if (i < 10) {
			file = path + "0" + to_string(i) + type;
		} else {
			file = path + to_string(i) + type;
		}
		ce.assign(file);
		p0 = ce.extract();
		ce.extract(p1, p2, p3);
		cout << p1 << p2 << p3 << endl;
	}
	return 0;
}
