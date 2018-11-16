/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void template_matching(int pos, void *userdata);

Mat input_image;
Mat template_image;

int match_method = 5;
int threshold_percent = 75;

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
//    imshow("Input Image", input_image);
    imshow("Template Image", template_image);

    namedWindow("Image display");
    createTrackbar(
            "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED",
            "Image display", &match_method, 5, template_matching);
    createTrackbar("In % of global value", "Image display", &threshold_percent, 100, template_matching);

    template_matching(0, NULL);

    // Sleep & do event loop.
    waitKey(0);

    return 0;
}

void template_matching(int pos, void *userdata)
{
    Mat result;
    matchTemplate(input_image, template_image, result, match_method);

    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    double minVal;
    double maxVal;
    minMaxLoc(result, &minVal, &maxVal);
    Mat mask = Mat::zeros(Size(input_image.cols, input_image.rows), CV_32FC1);
    inRange(result, maxVal * ((double)threshold_percent/100.0), maxVal, mask);
    mask.convertTo(mask, CV_8UC1);

//    if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) matchLoc = minLoc;
//    else matchLoc = maxLoc;

    Mat img_display(input_image.clone());

    vector<vector<Point> > contours;
    findContours(mask, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for (int i = 0; i < contours.size(); ++i)
    {
        vector<Point> hull;
        convexHull(contours[i], hull);
        Rect rect = boundingRect(hull);

        Point loc;
        minMaxLoc(result(rect), NULL, NULL, NULL, &loc);

        cout << loc << endl;

        Point c(loc.x + rect.x, loc.y + rect.y);
        rectangle(img_display, c, Point(c.x + template_image.cols, c.y + template_image.rows), Scalar::all(0), 2, 8, 0);
        rectangle(mask, rect.tl(), rect.br(), Scalar::all(50), 2, 8, 0);
    }

    imshow("Mask", mask);
    imshow("Image display", img_display);
    imshow("Result", result);
}
