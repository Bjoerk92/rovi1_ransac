#ifndef CIRC_EXTR_HELP_H
#define CIRC_EXTR_HELP_H
#include <opencv2/core/core.hpp>
#include <cmath>
using namespace std;
using namespace cv;

struct Circle_sizer {
	double s;
	int c;
	vector<Point> con;
	Point2f center;
	bool is_red;
	bool is_top_right;
	bool is_bottom_left;
	bool is_bottom_right;
	bool operator < (const Circle_sizer& rhs) const
	{
		return s < rhs.s;
	}
};

struct Circles{
	int red;
	int top_right;
	int bottom_left;
	int bottom_right;
};

#endif
