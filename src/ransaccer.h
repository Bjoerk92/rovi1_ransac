#ifndef RANSACCER_H
#define RANSACCER_H

#include <stdio.h>
#include <iostream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

class Ransac {
private:
    int _min_hessian;
    Point _center;
    double _max_dist;
    double _min_dist;
    Mat _img_object;
    Mat _img_scene;
    Mat _img_matches;
    Mat _H;
    vector<KeyPoint> _keypoints_object;
    vector<KeyPoint> _keypoints_scene;
    vector<Point2f> _obj;
    vector<Point2f> _scene;
    vector<Point2f> _obj_corners;
    vector<Point2f> _scene_corners;
    SurfDescriptorExtractor _extractor;
    Mat _descriptors_object;
	Mat _descriptors_scene;
    FlannBasedMatcher _matcher;
	vector<DMatch> _matches;
    vector<DMatch> _good_matches;
    void reset(void);
    void find_marker(void);
    void find_basic_object(void);
public:
    Ransac(void);
    void assign(Mat);
    void assign(String);
    Point extract(void);
    Mat draw_stuff(void);

};

#endif
