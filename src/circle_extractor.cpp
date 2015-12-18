#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include "circle_extractor.h"
#include "circ_extr_help.h"

using namespace std;
using namespace cv;

/* Public functions */

circle_extractor::circle_extractor(void)
{
	init_all();
}

circle_extractor::circle_extractor(Mat img)
{
	init_all();
	circle_extractor::assign(img);
}

Mat circle_extractor::get_image(int variant)
{
	Mat return_image;
	switch(variant) {
		case IMG_VAR_IMAGE:
		return_image = _image;
		break;
		case IMG_VAR_IMAGE_GRAY:
		return_image = _image_gray;
		break;
		case IMG_VAR_IMAGE_BINARY:
		return_image = _image_binary;
		break;
		case IMG_VAR_IMAGE_ERODED:
		return_image = _image_eroded;
		break;
		case IMG_VAR_IMAGE_OPENED:
		return_image = _image_opened;
		break;
		case IMG_VAR_IMAGE_CLEANED:
		return_image = _image_cleaned;
		break;
		case IMG_VAR_IMAGE_BLOBS:
		draw_blobs();
		return_image = _image_blobs;
		break;
		case IMG_VAR_MARKER:
		return_image = _org_marker;
		break;
		default:
		return_image = Mat::zeros(2,2, CV_8UC1);
		break;
	}
	return return_image.clone();
}

Point2f circle_extractor::extract()
{
	findContours(_image_cleaned, _contours, _hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	circle_extractor::find_circles();
	sort(_vec.begin(), _vec.end());

	for (int j{0}; j < 4; j++) {
		_cc[j] = _image.at<Vec3b>((_vec.end() - j - 1)->center);
	}

	circle_extractor::identify_circles();
	circle_extractor::centerize();

	return _marker_center;
}

void circle_extractor::extract(Point2f& p1, Point2f& p2, Point2f& p3)
{
	findContours(_image_cleaned, _contours, _hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	circle_extractor::find_circles();
	sort(_vec.begin(), _vec.end());

	for (int j{0}; j < 4; j++) {
		_cc[j] = _image.at<Vec3b>((_vec.end() - j - 1)->center);
	}

	circle_extractor::identify_circles();
	circle_extractor::centerize();
	p1 = _image_points[0];
	p2 = _image_points[1];
	p3 = _image_points[2];
}

void circle_extractor::assign(string image_name)
{
	circle_extractor::reset_some();
	_image = imread(image_name, CV_LOAD_IMAGE_COLOR);
	circle_extractor::prepare_image();
}

void circle_extractor::assign(Mat img)
{
	circle_extractor::reset_some();
	_image = img.clone();
	circle_extractor::prepare_image();
}


/* Private functions */

void circle_extractor::init_all(void)
{
	_circle_order = {-1, -1, -1, -1};
	_dilation_size = 2;
	_dilation_type = 2;
	_erode_type = 2;
	_erode_size = 3;
	_adaptive_thresh = 78;
	_adaptive_size = 85;
	_first_print = true;
	_dilate_element = getStructuringElement(_dilation_type,
			Size(2 * _dilation_size + 1, 2 * _dilation_size + 1),
		    Point(_dilation_size, _dilation_size));
	_erode_element = getStructuringElement(_erode_type,
			Size(2 * _erode_type + 1, 2 * _erode_type + 1),
			Point(_erode_size, _erode_size));
	_RED = 2;
	_vec.clear();
	_color = Scalar(255,0,0);
	_color_opposite = Scalar(0,255,0);
	_color_top_right = Scalar(0,255,255);
	_color_bottom_left = Scalar(255,255,0);
	init_org();
}

void circle_extractor::init_org(void)
{
	_org_marker = imread("../imgs/marker_1.png");
	cvtColor(_org_marker, _org_marker_manipulate, CV_RGB2GRAY);
	// adaptiveThreshold(_org_marker, _org_marker, _adaptive_thresh,
			// ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, _adaptive_size, 0);
	// imshow("init_org_test", _org_marker);
	// waitKey(0);
	// erode(_org_marker, _org_marker, _erode_element);
	// imshow("init_org_test", _org_marker);
	// waitKey(0);
	// dilate(_org_marker, _org_marker, _dilate_element);
	// imshow("init_org_test", _org_marker);
	// waitKey(0);
	threshold(_org_marker_manipulate, _org_marker_manipulate, 128, 255, THRESH_BINARY);

	findContours(_org_marker_manipulate, _org_contours, _org_hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	double area{0}, perimeter{0}, relation{0};
	Moments mu;
	Point mc;
	Circle_sizer to_push;
	for (unsigned int j{0}; j < _org_contours.size(); j++) {
		area = contourArea(_org_contours[j], false);

		if (area > 1000) {
			perimeter = arcLength(_org_contours[j], true);
			relation = 4 * M_PI * area / (perimeter * perimeter);
			mu = moments(_org_contours[j], true);
			mc = Point(mu.m10/mu.m00, mu.m01/mu.m00);
			to_push = {relation, static_cast<int>(j), _org_contours[j], mc, false, false, false, false};
			_org_vec.push_back(to_push);
		}
	}
	sort(_org_vec.begin(), _org_vec.end());
	for (int j{0}; j < 4; j++) {
		_org_cc[j] = _org_marker.at<Vec3b>((_org_vec.end() - j - 1)->center);
	}
	identify_org_circles();
}

void circle_extractor::centerize(void)
{
	_image_points[0] = (_vec.end() - _circle_order.red)->center;
	_image_points[1] = (_vec.end() - _circle_order.top_right)->center;
	_image_points[2] = (_vec.end() - _circle_order.bottom_left)->center;
	_image_points[3] = (_vec.end() - _circle_order.bottom_right)->center;
	_marker_points[0] = (_org_vec.end() - _org_circle_order.red)->center;
	_marker_points[1] = (_org_vec.end() - _org_circle_order.top_right)->center;
	_marker_points[2] = (_org_vec.end() - _org_circle_order.bottom_left)->center;
	_marker_points[3] = (_org_vec.end() - _org_circle_order.bottom_right)->center;

	_marker_center.x = _image_points[0].x + _image_points[1].x + _image_points[2].x + _image_points[3].x;
	_marker_center.y = _image_points[0].y + _image_points[1].y + _image_points[2].y + _image_points[3].y;
	_marker_center.x /= 4;
	_marker_center.y /= 4;
}

void circle_extractor::draw_blobs(void)
{
	// circle_extractor::centerize();
	_image_blobs = _image.clone();
	drawContours(_image_blobs, _contours, (_vec.end() - _circle_order.red         )->c, _color,             2, 8, _hierarchy, 0, Point(0, 0));
	drawContours(_image_blobs, _contours, (_vec.end() - _circle_order.bottom_right)->c, _color_opposite,    2, 8, _hierarchy, 0, Point(0, 0));
	drawContours(_image_blobs, _contours, (_vec.end() - _circle_order.top_right   )->c, _color_top_right,   2, 8, _hierarchy, 0, Point(0, 0));
	drawContours(_image_blobs, _contours, (_vec.end() - _circle_order.bottom_left )->c, _color_bottom_left, 2, 8, _hierarchy, 0, Point(0, 0));
	circle(_image_blobs, (_vec.end() - _circle_order.red         )->center, 4, _color,             -1, 8, 0);
	circle(_image_blobs, (_vec.end() - _circle_order.bottom_right)->center, 4, _color_opposite,    -1, 8, 0);
	circle(_image_blobs, (_vec.end() - _circle_order.top_right   )->center, 4, _color_top_right,   -1, 8, 0);
	circle(_image_blobs, (_vec.end() - _circle_order.bottom_left )->center, 4, _color_bottom_left, -1, 8, 0);
	circle(_image_blobs, _marker_center,                                    4, Scalar(0,0,0),      -1, 8, 0);
}

void circle_extractor::reset_some(void)
{
	_circle_order = {-1, -1, -1, -1};
	_first_print = true;
	_vec.clear();
	_contours.clear();
	_hierarchy.clear();
	_marker_center = Point2f(0,0);
	_image = Scalar(0,0,0);
	_image_binary = Scalar(0,0,0);
	_image_eroded = Scalar(0,0,0);
	_image_opened = Scalar(0,0,0);
	_image_cleaned = Scalar(0,0,0);
	_image_gray = Scalar(0,0,0);
	_image_blobs = Scalar(0,0,0);
}

void circle_extractor::find_circles(void)
{
	double area{0}, perimeter{0}, relation{0};
	Moments mu;
	Point mc;
	Circle_sizer to_push;
	for (unsigned int j{0}; j < _contours.size(); j++) {
		area = contourArea(_contours[j], false);

		if (area > 1000) {
			perimeter = arcLength(_contours[j], true);
			relation = 4 * M_PI * area / (perimeter * perimeter);
			mu = moments(_contours[j], true);
			mc = Point(mu.m10/mu.m00, mu.m01/mu.m00);
			to_push = {relation, static_cast<int>(j), _contours[j], mc, false, false, false, false};
			_vec.push_back(to_push);
		}
	}
}

void circle_extractor::identify_circles(void)
{
	int opposite{-1};
	if (       _cc[0].val[_RED] > _cc[1].val[_RED] and _cc[0].val[_RED] > _cc[2].val[_RED] and _cc[0].val[_RED] > _cc[3].val[_RED]) {
		(_vec.end() - 1)->is_red = true;
		_circle_order.red = 1;
		opposite = max_dist((_vec.end() - 1)->center, (_vec.end() - 2)->center, (_vec.end() - 3)->center, (_vec.end() - 4)->center);
		if (opposite == -1) {
			exit(-2);
		}
		(_vec.end() - opposite)->is_bottom_right = true;
		_circle_order.bottom_right = opposite;
	} else if (_cc[1].val[_RED] > _cc[0].val[_RED] and _cc[1].val[_RED] > _cc[2].val[_RED] and _cc[1].val[_RED] > _cc[3].val[_RED]) {
		(_vec.end() - 2)->is_red = true;
		_circle_order.red = 2;
		opposite = max_dist((_vec.end() - 2)->center, (_vec.end() - 1)->center, (_vec.end() - 3)->center, (_vec.end() - 4)->center);
		if (opposite == -1) {
			exit(-3);
		}
		switch(opposite) {
			case 2:
			(_vec.end() - 1)->is_bottom_right = true;
			_circle_order.bottom_right = 1;
			break;
			case 3:
			case 4:
			(_vec.end() - opposite)->is_bottom_right = true;
			_circle_order.bottom_right = opposite;
			break;
		}
	} else if (_cc[2].val[_RED] > _cc[0].val[_RED] and _cc[2].val[_RED] > _cc[1].val[_RED] and _cc[2].val[_RED] > _cc[3].val[_RED]) {
		(_vec.end() - 3)->is_red = true;
		_circle_order.red = 3;
		opposite = max_dist((_vec.end() - 3)->center, (_vec.end() - 1)->center, (_vec.end() - 2)->center, (_vec.end() - 4)->center);
		if (opposite == -1) {
			exit(-4);
		}
		switch(opposite) {
			case 2:
			case 3:
			(_vec.end() - (opposite - 1))->is_bottom_right = true;
			_circle_order.bottom_right = (opposite - 1);
			drawContours(_image_blobs, _contours, (_vec.end() - (opposite - 1))->c, _color_opposite, 2, 8, _hierarchy, 0, Point(0, 0));
			break;
			case 4:
			(_vec.end() - 4)->is_bottom_right = true;
			_circle_order.bottom_right = 4;
			break;
		}
	} else if (_cc[3].val[_RED] > _cc[0].val[_RED] and _cc[3].val[_RED] > _cc[1].val[_RED] and _cc[3].val[_RED] > _cc[2].val[_RED]) {
		(_vec.end() - 4)->is_red = true;
		_circle_order.red = 4;
		opposite = max_dist((_vec.end() - 4)->center, (_vec.end() - 1)->center, (_vec.end() - 2)->center, (_vec.end() - 3)->center);
		if (opposite == -1) {
			exit(-5);
		}
		(_vec.end() - (opposite -1))->is_bottom_right = true;
		_circle_order.bottom_right = (opposite -1);
	}

	int temp1, temp2;

	for (int i{1}; i < 5; i++) {
		if (_circle_order.red != i and _circle_order.bottom_right != i) {
			temp1 = i;
		}
	}
	for (int i{1}; i < 5; i++) {
		if (_circle_order.red != i and _circle_order.bottom_right != i and temp1 != i) {
			temp2 = i;
			break;
		}
	}
	//R.
	//.X
	if (    (_vec.end() - _circle_order.red)->center.x < (_vec.end() - _circle_order.bottom_right)->center.x
		and (_vec.end() - _circle_order.red)->center.y < (_vec.end() - _circle_order.bottom_right)->center.y) {
		if ((_vec.end() - temp1)->center.y < (_vec.end() - temp2)->center.y) {
			_circle_order.top_right = temp1;
			(_vec.end() - temp1)->is_top_right = true;
			_circle_order.bottom_left = temp2;
			(_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_circle_order.top_right = temp2;
			(_vec.end() - temp2)->is_top_right = true;
			_circle_order.bottom_left = temp1;
			(_vec.end() - temp1)->is_bottom_left = true;
		}
	}

	//.R
	//X.
	else if ((_vec.end() - _circle_order.red)->center.x > (_vec.end() - _circle_order.bottom_right)->center.x
		and  (_vec.end() - _circle_order.red)->center.y < (_vec.end() - _circle_order.bottom_right)->center.y) {
		if ((_vec.end() - temp1)->center.x > (_vec.end() - temp2)->center.x) {
			_circle_order.top_right = temp1;
			(_vec.end() - temp1)->is_top_right = true;
			_circle_order.bottom_left = temp2;
			(_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_circle_order.top_right = temp2;
			(_vec.end() - temp2)->is_top_right = true;
			_circle_order.bottom_left = temp1;
			(_vec.end() - temp1)->is_bottom_left = true;
		}
	}

	//X.
	//.R
	else if ((_vec.end() - _circle_order.red)->center.x > (_vec.end() - _circle_order.bottom_right)->center.x
		and  (_vec.end() - _circle_order.red)->center.y > (_vec.end() - _circle_order.bottom_right)->center.y) {
		if ((_vec.end() - temp1)->center.x < (_vec.end() - temp2)->center.x) {
			_circle_order.top_right = temp1;
			(_vec.end() - temp1)->is_top_right = true;
			_circle_order.bottom_left = temp2;
			(_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_circle_order.top_right = temp2;
			(_vec.end() - temp2)->is_top_right = true;
			_circle_order.bottom_left = temp1;
			(_vec.end() - temp1)->is_bottom_left = true;
		}
	}

	//.X
	//R.
	else if ((_vec.end() - _circle_order.red)->center.x < (_vec.end() - _circle_order.bottom_right)->center.x
		and  (_vec.end() - _circle_order.red)->center.y > (_vec.end() - _circle_order.bottom_right)->center.y) {
		if ((_vec.end() - temp1)->center.y < (_vec.end() - temp2)->center.y) {
			_circle_order.top_right = temp1;
			(_vec.end() - temp1)->is_top_right = true;
			_circle_order.bottom_left = temp2;
			(_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_circle_order.top_right = temp2;
			(_vec.end() - temp2)->is_top_right = true;
			_circle_order.bottom_left = temp1;
			(_vec.end() - temp1)->is_bottom_left = true;
		}
	}
}

void circle_extractor::identify_org_circles(void)
{
	int opposite{-1};
	if (       _org_cc[0].val[_RED] > _org_cc[1].val[_RED] and _org_cc[0].val[_RED] > _org_cc[2].val[_RED] and _org_cc[0].val[_RED] > _org_cc[3].val[_RED]) {
		(_org_vec.end() - 1)->is_red = true;
		_org_circle_order.red = 1;
		opposite = max_dist((_org_vec.end() - 1)->center, (_org_vec.end() - 2)->center, (_org_vec.end() - 3)->center, (_org_vec.end() - 4)->center);
		if (opposite == -1) {
			exit(-2);
		}
		(_org_vec.end() - opposite)->is_bottom_right = true;
		_org_circle_order.bottom_right = opposite;
	} else if (_org_cc[1].val[_RED] > _org_cc[0].val[_RED] and _org_cc[1].val[_RED] > _org_cc[2].val[_RED] and _org_cc[1].val[_RED] > _org_cc[3].val[_RED]) {
		(_org_vec.end() - 2)->is_red = true;
		_org_circle_order.red = 2;
		opposite = max_dist((_org_vec.end() - 2)->center, (_org_vec.end() - 1)->center, (_org_vec.end() - 3)->center, (_org_vec.end() - 4)->center);
		if (opposite == -1) {
			exit(-3);
		}
		switch(opposite) {
			case 2:
			(_org_vec.end() - 1)->is_bottom_right = true;
			_org_circle_order.bottom_right = 1;
			break;
			case 3:
			case 4:
			(_org_vec.end() - opposite)->is_bottom_right = true;
			_org_circle_order.bottom_right = opposite;
			break;
		}
	} else if (_org_cc[2].val[_RED] > _org_cc[0].val[_RED] and _org_cc[2].val[_RED] > _org_cc[1].val[_RED] and _org_cc[2].val[_RED] > _org_cc[3].val[_RED]) {
		(_org_vec.end() - 3)->is_red = true;
		_org_circle_order.red = 3;
		opposite = max_dist((_org_vec.end() - 3)->center, (_org_vec.end() - 1)->center, (_org_vec.end() - 2)->center, (_org_vec.end() - 4)->center);
		if (opposite == -1) {
			exit(-4);
		}
		switch(opposite) {
			case 2:
			case 3:
			(_org_vec.end() - (opposite - 1))->is_bottom_right = true;
			_org_circle_order.bottom_right = (opposite - 1);
			drawContours(_image_blobs, _contours, (_org_vec.end() - (opposite - 1))->c, _color_opposite, 2, 8, _hierarchy, 0, Point(0, 0));
			break;
			case 4:
			(_org_vec.end() - 4)->is_bottom_right = true;
			_org_circle_order.bottom_right = 4;
			break;
		}
	} else if (_org_cc[3].val[_RED] > _org_cc[0].val[_RED] and _org_cc[3].val[_RED] > _org_cc[1].val[_RED] and _org_cc[3].val[_RED] > _org_cc[2].val[_RED]) {
		(_org_vec.end() - 4)->is_red = true;
		_org_circle_order.red = 4;
		opposite = max_dist((_org_vec.end() - 4)->center, (_org_vec.end() - 1)->center, (_org_vec.end() - 2)->center, (_org_vec.end() - 3)->center);
		if (opposite == -1) {
			exit(-5);
		}
		(_org_vec.end() - (opposite -1))->is_bottom_right = true;
		_org_circle_order.bottom_right = (opposite -1);
	}

	int temp1, temp2;

	for (int i{1}; i < 5; i++) {
		if (_org_circle_order.red != i and _org_circle_order.bottom_right != i) {
			temp1 = i;
		}
	}
	for (int i{1}; i < 5; i++) {
		if (_org_circle_order.red != i and _org_circle_order.bottom_right != i and temp1 != i) {
			temp2 = i;
			break;
		}
	}

	//R.
	//.X
	if (    (_org_vec.end() - _org_circle_order.red)->center.x < (_org_vec.end() - _org_circle_order.bottom_right)->center.x
		and (_org_vec.end() - _org_circle_order.red)->center.y < (_org_vec.end() - _org_circle_order.bottom_right)->center.y) {
		if ((_org_vec.end() - temp1)->center.y < (_org_vec.end() - temp2)->center.y) {
			_org_circle_order.top_right = temp1;
			(_org_vec.end() - temp1)->is_top_right = true;
			_org_circle_order.bottom_left = temp2;
			(_org_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_org_circle_order.top_right = temp2;
			(_org_vec.end() - temp2)->is_top_right = true;
			_org_circle_order.bottom_left = temp1;
			(_org_vec.end() - temp1)->is_bottom_left = true;
		}
	}

	//.R
	//X.
	else if ((_org_vec.end() - _org_circle_order.red)->center.x > (_org_vec.end() - _org_circle_order.bottom_right)->center.x
		and  (_org_vec.end() - _org_circle_order.red)->center.y < (_org_vec.end() - _org_circle_order.bottom_right)->center.y) {
		if ((_org_vec.end() - temp1)->center.x > (_org_vec.end() - temp2)->center.x) {
			_org_circle_order.top_right = temp1;
			(_org_vec.end() - temp1)->is_top_right = true;
			_org_circle_order.bottom_left = temp2;
			(_org_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_org_circle_order.top_right = temp2;
			(_org_vec.end() - temp2)->is_top_right = true;
			_org_circle_order.bottom_left = temp1;
			(_org_vec.end() - temp1)->is_bottom_left = true;
		}
	}

	//X.
	//.R
	else if ((_org_vec.end() - _org_circle_order.red)->center.x > (_org_vec.end() - _org_circle_order.bottom_right)->center.x
		and  (_org_vec.end() - _org_circle_order.red)->center.y > (_org_vec.end() - _org_circle_order.bottom_right)->center.y) {
		if ((_org_vec.end() - temp1)->center.x < (_org_vec.end() - temp2)->center.x) {
			_org_circle_order.top_right = temp1;
			(_org_vec.end() - temp1)->is_top_right = true;
			_org_circle_order.bottom_left = temp2;
			(_org_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_org_circle_order.top_right = temp2;
			(_org_vec.end() - temp2)->is_top_right = true;
			_org_circle_order.bottom_left = temp1;
			(_org_vec.end() - temp1)->is_bottom_left = true;
		}
	}

	//.X
	//R.
	else if ((_org_vec.end() - _org_circle_order.red)->center.x < (_org_vec.end() - _org_circle_order.bottom_right)->center.x
		and  (_org_vec.end() - _org_circle_order.red)->center.y > (_org_vec.end() - _org_circle_order.bottom_right)->center.y) {
		if ((_org_vec.end() - temp1)->center.y < (_org_vec.end() - temp2)->center.y) {
			_org_circle_order.top_right = temp1;
			(_org_vec.end() - temp1)->is_top_right = true;
			_org_circle_order.bottom_left = temp2;
			(_org_vec.end() - temp2)->is_bottom_left = true;
		} else {
			_org_circle_order.top_right = temp2;
			(_org_vec.end() - temp2)->is_top_right = true;
			_org_circle_order.bottom_left = temp1;
			(_org_vec.end() - temp1)->is_bottom_left = true;
		}
	}
}

void circle_extractor::prepare_image(void)
{
	cvtColor(_image, _image_gray, CV_RGB2GRAY);
	adaptiveThreshold(_image_gray, _image_binary, _adaptive_thresh,
			ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, _adaptive_size, 0);
	erode(_image_binary, _image_eroded, _erode_element);
	dilate(_image_eroded, _image_opened, _dilate_element);
	threshold(_image_opened, _image_cleaned, 50, 255, THRESH_BINARY);
}

int circle_extractor::max_dist(Point2f red, Point2f first, Point2f second, Point2f third)
{
	int return_value{-1};

	double dist0 = dist(red, first);
	double dist1 = dist(red, second);
	double dist2 = dist(red, third);

	if (dist0 > dist1 and dist0 > dist2) {
		return_value = 2;
	} else if (dist1 > dist0 and dist1 > dist2) {
		return_value = 3;
	} else if (dist2 > dist0 and dist2 > dist1) {
		return_value = 4;
	}

	return return_value;
}

double circle_extractor::dist(Point2f first, Point2f second)
{
	return sqrt((first.x - second.x) * (first.x - second.x) + (first.y - second.y) * (first.y - second.y));
}
