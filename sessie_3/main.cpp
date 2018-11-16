/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void run(int pos, void *userdata);

Mat input_image;
Mat template_image;
Point center;

int match_method = 5;
int threshold_percent = 50;
int rotation_steps = 19;
int rotation_max = 360;

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<file>          |     | Input image. }"
                             "{ @<template>      |     | Template image. }"
    );
    parser.about("Parameters surrounded by <> are required.\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string file_name(parser.get<string>("@<file>"));
    string template_Name(parser.get<string>("@<template>"));
    // Required arguments check
    if (file_name.empty() || template_Name.empty())
    {
        cout << "Missing arguments." << endl;
        parser.printMessage();
        return -1;
    }

    // Load images & error check
    input_image = imread(file_name, IMREAD_COLOR);
    template_image = imread(template_Name, IMREAD_COLOR);

    if (!input_image.data || !template_image.data)
    {
        cout << "Could not load files." << endl;
        return -1;
    }

    // Display image as loaded.
//    imshow("Template Image", template_image);

//    namedWindow("Sliders", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    namedWindow("Image display", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    createTrackbar("Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED", "Image display", &match_method, 5, run);
    createTrackbar("Threshold(% of max)", "Image display", &threshold_percent, 100, run);
    createTrackbar("Rotation steps", "Image display", &rotation_steps, 36, run);
    createTrackbar("Rotation max", "Image display", &rotation_max, 360, run);

//    int radius = (int)(max(input_image.cols, input_image.rows))/2;//(int) sqrt(input_image.cols*input_image.cols + input_image.rows*input_image.rows)/2;
//    cout << "Size: " << input_image.size() << " Offset: " << radius << endl;
//    center.x = input_image.cols / 2;
//    center.y = input_image.rows / 2;
//    copyMakeBorder(input_image.clone(), input_image, radius-center.y, radius-center.y, radius-center.x, radius-center.x, BORDER_CONSTANT);
    center.x = input_image.cols / 2;
    center.y = input_image.rows / 2;

    run(0, NULL);

    // Sleep & do event loop.
    waitKey(0);

    return 0;
}

vector<vector<Point> > match(Mat input_image)
{
    Mat result;
    matchTemplate(input_image, template_image, result, match_method);
//    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

    if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
        result = 1 - result;

//    double maxVal;
//    minMaxLoc(result, NULL, &maxVal);
    Mat mask = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC1);
    inRange(result, ((double)threshold_percent/100.0), 1, mask);

    vector<vector<Point> > corners;
    vector<vector<Point> > contours;
    findContours(mask, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (int i = 0; i < contours.size(); ++i)
    {
        vector<Point> hull;
        convexHull(contours[i], hull);
        Rect rect = boundingRect(hull);
//        rectangle(mask, rect.tl(), rect.br(), Scalar::all(50), 2, 8, 0);

        Point loc;
        minMaxLoc(result(rect), NULL, NULL, NULL, &loc);

        Point c(loc.x + rect.x, loc.y + rect.y);

        vector<Point> points;
        points.push_back(c);
        points.push_back(c + Point(0, template_image.rows));
        points.push_back(c + Point(template_image.cols, template_image.rows));
        points.push_back(c + Point(template_image.cols, 0));
        corners.push_back(points);
    }

//    imshow("Mask", mask);
//    imshow("Image display", img_display);
//    imshow("Result", result);

    return corners;
}

void drawBoxes(Mat image, vector<vector<Point> > corners, Scalar color, double angle, Point center, Mat no_rot)
{
    Mat rotation = getRotationMatrix2D(center, -angle, 1.0);

    for (int i = 0; i < corners.size(); ++i)
    {
        vector<Point> p = corners[i];

//        cout << "Non rotated: " << p << endl;
//        {
//            Point prev = p[0];
//            for (int j = 1; j < p.size(); j++)
//            {
//                line(no_rot, prev, p[j], color, 2);
//                prev = p[j];
//            }
//            line(no_rot, prev, p[0], color, 2);
//        }

        Mat_<double>m(3,4);
        for (int j = 0; j < p.size(); j++)
        {
            m.at<double>(0, j) = (double) p[j].x;
            m.at<double>(1, j) = (double) p[j].y;
            m.at<double>(2, j) = 1;
        }

        m = rotation * m;

        for (int j = 0; j < p.size(); j++)
        {
            p[j].x = (int)m.at<double>(0, j);
            p[j].y = (int)m.at<double>(1, j);
        }

//        cout << "Rotated: " << p << endl;

        {
            Point prev = p[0];
            for (int j = 1; j < p.size(); j++)
            {
                line(image, prev, p[j], color, 2);
                prev = p[j];
            }
            line(image, prev, p[0], color, 2);
        }
    }
}

void run(int pos, void *userdata)
{
    if (rotation_steps == 0 || rotation_max == 0)
    {
        Mat image(input_image.clone());
        vector<vector<Point> > corners = match(image);
        for (int i = 0; i < corners.size(); ++i)
            rectangle(image, corners[i][0], corners[i][2], Scalar::all(0));
        imshow("Image display", image);
        return;
    }

    Mat output(input_image.clone());

    cout << "Rotating in steps of " << (double)rotation_max/(double)rotation_steps << "°" << endl;
    for (int i = 0; i < rotation_steps; i++)
    {
        double angle = i*((double)rotation_max/(double)rotation_steps);
//        cout << angle << "°" << endl;
        Mat rot_image;
        Mat rotation = getRotationMatrix2D(center, angle, 1.0);
        warpAffine(input_image, rot_image, rotation, input_image.size());

        vector<vector<Point> > corners = match(rot_image);
        drawBoxes(output, corners, Scalar(244, 66, 232), angle, center, rot_image);

//        Mat smaller;
//        resize(rot_image, smaller, Size(max_dim/2, max_dim/2));
//        imshow("Rotated image", smaller);
//        Mat smaller2;
//        resize(output, smaller2, Size(max_dim/2, max_dim/2));
//        imshow("Image display", smaller2);
//        waitKey(0);
    }

//    Mat smaller2;
//    resize(output, smaller2, Size(max_dim/2, max_dim/2));
    imshow("Image display", output);
}
