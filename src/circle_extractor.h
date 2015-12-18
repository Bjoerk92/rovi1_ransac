#ifndef CIRCLE_EXTRACTOR_H
#define CIRCLE_EXTRACTOR_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <array>
#include "circ_extr_help.h"

#define IMG_VAR_IMAGE			0
#define IMG_VAR_IMAGE_GRAY		1
#define IMG_VAR_IMAGE_BINARY	2
#define IMG_VAR_IMAGE_ERODED	3
#define IMG_VAR_IMAGE_OPENED	4
#define IMG_VAR_IMAGE_CLEANED	5
#define IMG_VAR_IMAGE_BLOBS		6
#define IMG_VAR_MARKER			7

using namespace std;
using namespace cv;

class circle_extractor{
private:
	int _RED;
	Circles _circle_order;
	int _dilation_size, _dilation_type;
	int _erode_type, _erode_size;
	Mat _dilate_element, _erode_element;
	int _adaptive_thresh, _adaptive_size;
	Point2f _marker_center;

	bool _first_print;
	Mat _image, _image_gray, _image_binary, _image_eroded, _image_opened, _image_cleaned, _image_blobs;

	Mat _org_marker, _org_marker_manipulate, _org_marker_blobs;
	vector<Circle_sizer> _org_vec;
	vector<vector<Point> > _org_contours;
	vector<Vec4i> _org_hierarchy;
	Circles _org_circle_order;
	Vec3b _org_cc[4];

	Point2f _marker_points[4];
	Point2f _image_points[4];

	Scalar _color;
	Scalar _color_opposite;
	Scalar _color_top_right;
	Scalar _color_bottom_left;
	Vec3b _cc[4];
	vector<Circle_sizer> _vec;
	vector<vector<Point> > _contours;
	vector<Vec4i> _hierarchy;
	void find_circles(void);
	void identify_circles(void);
	void identify_org_circles(void);
	void prepare_image(void);
	int max_dist(Point2f, Point2f, Point2f, Point2f);
	double dist(Point2f, Point2f);
	void print_circle_order(void);
	void init_all(void);
	void init_org(void);
	void reset_some(void);
	void draw_blobs(void);
	void centerize(void);

public:
	circle_extractor(void);
	circle_extractor(Mat);
	Point2f extract(void);
	void extract(Point2f&, Point2f&, Point2f&);
	void assign(string);
	void assign(Mat);
	Mat get_image(int);
};
#endif
