#include "ransaccer.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

Ransac::Ransac(void)
{
    _img_object = imread("../src/Corny.png", CV_LOAD_IMAGE_COLOR);
    _min_hessian = 400;
    _max_dist = 0;
    _min_dist = 1000;
    _obj_corners.resize(4);
    _scene_corners.resize(4);
    _center.x = 0;
    _center.y = 0;
    Ransac::find_basic_object();
}

void Ransac::reset(void)
{
    _max_dist = 0;
    _min_dist = 1000;
    _scene_corners.clear();
    _keypoints_scene.clear();
    _obj.clear();
    _scene.clear();
    _matches.clear();
    _good_matches.clear();
    _center.x = 0;
    _center.y = 0;
}

void Ransac::assign(Mat img)
{
    Ransac::reset();
    _img_scene = img.clone();
    Ransac::find_marker();
}

void Ransac::assign(String name)
{
    Ransac::reset();
    _img_scene = imread(name, CV_LOAD_IMAGE_COLOR);
    Ransac::find_marker();
}

Point Ransac::extract(void)
{
    return _center;
}

void Ransac::find_basic_object(void)
{
    SurfFeatureDetector detector(_min_hessian);
    detector.detect(_img_object, _keypoints_object);
    _extractor.compute(_img_object, _keypoints_object, _descriptors_object);
    _obj_corners[0] = cvPoint(0,0);
    _obj_corners[1] = cvPoint(_img_object.cols, 0);
    _obj_corners[2] = cvPoint(_img_object.cols, _img_object.rows);
    _obj_corners[3] = cvPoint(0, _img_object.rows);
}

void Ransac::find_marker(void)
{
    SurfFeatureDetector detector(_min_hessian);
    detector.detect(_img_scene, _keypoints_scene);
    _extractor.compute(_img_scene, _keypoints_scene, _descriptors_scene);
    _matcher.match(_descriptors_object, _descriptors_scene, _matches);
    for (int i{0}; i < _descriptors_object.rows; i++) {
        double dist = _matches[i].distance;
        if (dist < _min_dist ) {
            _min_dist = dist;
        } else if (dist > _max_dist) {
            _max_dist = dist;
        }
    }
    for (int i{0}; i < _descriptors_object.rows; i++) {
        if (_matches[i].distance < 3 * _min_dist) {
            _good_matches.push_back(_matches[i]);
        }
    }

    for (int i{0}; i < _good_matches.size(); i++) {
        _obj.push_back(_keypoints_object[_good_matches[i].queryIdx].pt);
        _scene.push_back(_keypoints_scene[_good_matches[i].trainIdx].pt);
    }
    _H = findHomography(_obj, _scene, CV_RANSAC);
    perspectiveTransform(_obj_corners, _scene_corners, _H);

    for (int i{0}; i < 4; i++) {
        _center.x = _center.x + (_scene_corners[i].x + _img_object.cols);
        _center.y = _center.y + _scene_corners[i].y;
    }

    _center.x = _center.x/4;
    _center.y = _center.y/4;
}

Mat Ransac::draw_stuff(void)
{
    drawMatches(_img_object, _keypoints_object, _img_scene, _keypoints_scene,
                _good_matches, _img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    line(_img_matches, _scene_corners[0] + Point2f(_img_object.cols, 0), _scene_corners[1] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(_img_matches, _scene_corners[1] + Point2f(_img_object.cols, 0), _scene_corners[2] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(_img_matches, _scene_corners[2] + Point2f(_img_object.cols, 0), _scene_corners[3] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(_img_matches, _scene_corners[3] + Point2f(_img_object.cols, 0), _scene_corners[0] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);

    circle(_img_matches, Point(_center.x,_center.y) , 4, Scalar(0,255,0), 5);
    #ifdef DEBUG

    imshow("Center of image", _img_matches);
    waitKey(0);
    #endif
    return _img_matches.clone();
}
