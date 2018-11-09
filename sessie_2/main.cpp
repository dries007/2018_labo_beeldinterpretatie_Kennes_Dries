/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat input_image; // NOLINT(cert-err58-cpp)

void on_trackbar(int, void*);

int R_MAX = 255;
int R_MIN = 125;
int G_MAX = 90;
int G_MIN = 0;
int B_MAX = 90;
int B_MIN = 0;
int H_MAX = 15;
int H_MIN = 160;
int S_MAX = 255;
int S_MIN = 110;
int V_MAX = 255;
int V_MIN = 110;


int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<file>          |     | Input file. }"
    );
    parser.about("Parameters surrounded by <> are required.\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string file(parser.get<string>("@<file>"));
    // Required arguments check
    if (file.empty())
    {
        cout << "Missing arguments." << endl;
        parser.printMessage();
        return -1;
    }

    // Load images & error check
    input_image = imread(file, 1);

    if (!input_image.data)
    {
        cout << "Could not load file." << endl;
        return -1;
    }

    // Display image as loaded.
    imshow("Input Image", input_image);

    namedWindow("Sliders", 1);

    createTrackbar("R_MAX", "Sliders", &R_MAX, 255, on_trackbar);
    createTrackbar("R_MIN", "Sliders", &R_MIN, 255, on_trackbar);
    createTrackbar("G_MAX", "Sliders", &G_MAX, 255, on_trackbar);
    createTrackbar("G_MIN", "Sliders", &G_MIN, 255, on_trackbar);
    createTrackbar("B_MAX", "Sliders", &B_MAX, 255, on_trackbar);
    createTrackbar("B_MIN", "Sliders", &B_MIN, 255, on_trackbar);
    createTrackbar("H_MAX", "Sliders", &H_MAX, 180, on_trackbar);
    createTrackbar("H_MIN", "Sliders", &H_MIN, 180, on_trackbar);
    createTrackbar("S_MAX", "Sliders", &S_MAX, 255, on_trackbar);
    createTrackbar("S_MIN", "Sliders", &S_MIN, 255, on_trackbar);
    createTrackbar("V_MAX", "Sliders", &V_MAX, 255, on_trackbar);
    createTrackbar("V_MIN", "Sliders", &V_MIN, 255, on_trackbar);

    on_trackbar(0, 0);

    // Sleep & do event loop.
    waitKey(0);

    return 0;
}

void on_trackbar(int, void*)
{
    // RGB masking
    Mat rgb_mask = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC1);
    inRange(input_image, Scalar(B_MIN, G_MIN, R_MIN), Scalar(B_MAX, G_MAX, R_MAX), rgb_mask);

    // Nuke noise
    erode(rgb_mask, rgb_mask, Mat(), Point(-1, -1), 1);
    dilate(rgb_mask, rgb_mask, Mat(), Point(-1, -1), 1);
    // Glue blobs
    dilate(rgb_mask, rgb_mask, Mat(), Point(-1, -1), 2);
    erode(rgb_mask, rgb_mask, Mat(), Point(-1, -1), 2);

    dilate(rgb_mask, rgb_mask, Mat(), Point(-1, -1), 10);

    Mat rgb_masked;
    input_image.copyTo(rgb_masked, rgb_mask);

    imshow("RGB Masked", rgb_masked);

    // HSV
    Mat hsv = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC3);
    cvtColor(input_image, hsv, COLOR_BGR2HSV);

    imshow("HSV", hsv);

    // HSV masking
    Mat hsv_mask = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC1);
    if (H_MAX < H_MIN)
    {

        Mat hsv_mask_1 = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC1);
        Mat hsv_mask_2 = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC1);
        inRange(hsv, Scalar(0, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hsv_mask_1);
        inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(180, S_MAX, V_MAX), hsv_mask_2);
        hsv_mask = hsv_mask_1 | hsv_mask_2;
    }
    else
    {
        inRange(hsv, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), hsv_mask);
    }

    // Nuke noise
    erode(hsv_mask, hsv_mask, Mat(), Point(-1, -1), 1);
    dilate(hsv_mask, hsv_mask, Mat(), Point(-1, -1), 1);
    // Glue blobs
    dilate(hsv_mask, hsv_mask, Mat(), Point(-1, -1), 5);
    erode(hsv_mask, hsv_mask, Mat(), Point(-1, -1), 5);

    dilate(rgb_mask, rgb_mask, Mat(), Point(-1, -1), 10);

    Mat hsv_masked;
    hsv.copyTo(hsv_masked, hsv_mask);

//    imshow("HSV Mask", hsv_mask);
//    imshow("HSV Masked", hsv_masked);

    Mat hsv_bgr = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC3);
    cvtColor(hsv_masked, hsv_bgr, COLOR_HSV2BGR);

    imshow("HSV2BGR Masked", hsv_bgr);

    // Contouring
    vector<vector<Point> > contours;
    findContours(hsv_mask, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<vector<Point> > hulls;
    int biggest_index = -1;
    double biggest_area = -1;
    for (int i = 0; i < contours.size(); ++i)
    {
        vector<Point> hull;
        convexHull(contours[i], hull);
        hulls.push_back(hull);
        double a = contourArea(hull);
        if (a > biggest_area)
        {
            biggest_area = a;
            biggest_index = i;
        }
    }

    if (biggest_index == -1)
        return;

    Mat hsv_mask_2 = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC1);
//    drawContours(hsv_mask_2, hulls, -1, 255, -1);

    vector<vector<Point> > hulls_tmp;
    hulls_tmp.push_back(hulls.at(biggest_index));
    drawContours(hsv_mask_2, hulls_tmp, -1, 255, -1);

    Mat hsv_masked_contoured;
    hsv.copyTo(hsv_masked_contoured, hsv_mask_2);

//    imshow("HSV Mask", hsv_mask);
//    imshow("HSV Masked", hsv_masked);

    Mat hsv_bgr_contoured = Mat::zeros(Size(input_image.cols, input_image.rows), CV_8UC3);
    cvtColor(hsv_masked_contoured, hsv_bgr_contoured, COLOR_HSV2BGR);

    imshow("HSV2BGR Masked Contoured", hsv_bgr_contoured);
}
