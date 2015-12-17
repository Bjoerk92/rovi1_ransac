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

ransaccer::ransaccer()
{
    _img_object = imread("../src/Corny.png", 1);
    _min_hessian = 400;
    _max_dist = 0;
    _min_dist = 100;
    _object_corners.resize(4, 0);
    _scene_corners.resize(4, 0);
    _x = 0;
    _y = 0;
}

void ransaccer::reset(void)
{
    _max_dist = 0;
    _min_dist = 100;
    _object_corners.clear();
    _scene_corners.clear();
    _x = 0;
    _y = 0;
}
void ransaccer::assign(Mat img)
{
    _img_scene = img.clone();
    ransaccer::find_marker();
}

void ransaccer::assign(String name)
{
    _img_scene = imread(name, CV_LOAD_IMAGE_COLOR);
    ransaccer::find_marker();
}
Point ransaccer::extract(void)
{
    return _center;
}

void find_marker(void)
{
    SurfFeatureDetector detector(_min_hessian);
    detector.detect(_img_object, _keypoints_object);
    detector.detect(_img_scene, _keypoints_scene);
    _extractor.compute(_img_object, _keypoints_object, _descriptors_object);
    _extractor.compute(_img_scene, _keypoints_scene, _descriptors_scene);
    _matcher.match(_descriptors_object, _descriptors_scene, _matches);

    for (int i{0}; i < _descriptors_object.rows; i++) {
        double dist = _matches[i].distance;
        if( _matches[i].distance < _min_dist ) {
            _min_dist = _matches[i].distance;
        } else if (_matches[i].distance > _max_dist) {
            _max_dist = _matches[i].distance;
        }
    }

    for (int i{0}; i < descriptors_object.rows; i++) {
        if (_matches[i].distance < 3 * _min_dist) {
            _good_matches.push_back(_matches[i]);
        }
    }
    #ifdef DEBUG
    drawMatches(_img_object, _keypoints_object, _img_scene, _keypoints_scene,
                _good_matches, _img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    #endif

    for (int i{0}; i < _good_matches.size(); i++) {
        _obj.push_back(_keypoints_object[_good_matches[i].queryIdx].pt);
        _scene.push_back(_keypoints_scene[_good_matches[i].trainIdx].pt);
    }

    _H = findHomography(_obj, _scene, CV_RANSAC);
    _obj_corners[0] = cvPoint(0,0);
    _obj_corners[1] = cvPoint(_img_object.cols, 0);
    _obj_corners[2] = cvPoint(_img_object.cols, _img_object.rows);
    _obj_corners[3] = cvPoint(0, _img_object.rows);
    perspectiveTransform(_obj_corners, _scene_corners, _H);

    #ifdef DEBUG
    line(_img_matches, _scene_corners[0] + Point2f(_img_object.cols, 0), _scene_corners[1] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(_img_matches, _scene_corners[1] + Point2f(_img_object.cols, 0), _scene_corners[2] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(_img_matches, _scene_corners[2] + Point2f(_img_object.cols, 0), _scene_corners[3] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(_img_matches, _scene_corners[3] + Point2f(_img_object.cols, 0), _scene_corners[0] + Point2f(_img_object.cols, 0), Scalar(0, 255, 0), 4);
    #endif

    for (int i{0}; i < 4; i++) {
        _center.x += _scene_corners[i].x;
        _center.y += _scene_corners[i].y;
    }

    _center.x /= 4;
    _center.y /= 4;

    #ifdef DEBUG
    circle(_img_scene, Point(_center.x,_center.y) , 4, Scalar(0,255,0), 5);
    imshow("Center of image", _img_scene);
    waitKey(0);
    #endif
}
