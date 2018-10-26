/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<file1>         |     | File 1, for skin segmentation. }"
                             "{ @<file2>         |     | File 2, for text segmentation. }"
    );
    parser.about("Parameters surrounded by <> are required.\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string file1(parser.get<string>("@<file1>"));
    string file2(parser.get<string>("@<file2>"));

    // Required arguments check
    if (file1.empty() || file2.empty())
    {
        cout << "Missing arguments." << endl;
        parser.printMessage();
        return -1;
    }

    // Load images & error check
    Mat image1 = imread(file1, IMREAD_COLOR); // Skin segmentation input
    Mat image2 = imread(file2, IMREAD_GRAYSCALE); // Text segmentation input

    if (!image1.data)
    {
        cout << "Could not load file 1: " << file1 << endl;
        return -1;
    }
    if (!image2.data)
    {
        cout << "Could not load file 2: " << file2 << endl;
        return -1;
    }

    resize(image2, image2, Size(image2.cols / 2, image2.rows / 2));

    // Image 1 processing
    // ------------------
    // PART 1

    // Split color image into RGB
    Mat image1_bgr[3];
    split(image1, image1_bgr);
    Mat r = image1_bgr[2], g = image1_bgr[1], b = image1_bgr[0];

    // Mask output image 1
    Mat image1_mask = Mat::zeros(image1.rows, image1.cols, CV_8UC1);
    image1_mask = ((r > 95) & (g > 40) & (b > 20) & ((max(r, max(g, b)) - min(r, min(g, b))) > 15) & (abs(r - g) > 15) & (r > g) & (r > g)) * 0xFF;

    // EZ-merge
    Mat image1_bgr_merge_p1[] = {b & image1_mask, g & image1_mask, r & image1_mask};
    Mat image1_skin_p1;
    merge(image1_bgr_merge_p1, 3, image1_skin_p1);
    //cout << image1_skin << endl;

    namedWindow("Display Image 1 (Skin Segmentation) part 1", WINDOW_AUTOSIZE);
    imshow("Display Image 1 (Skin Segmentation) part 1", image1_skin_p1);

    // PART 2

    // Nuke ruis
    erode(image1_mask, image1_mask, Mat(), Point(-1, -1), 2);
    dilate(image1_mask, image1_mask, Mat(), Point(-1, -1), 2);

    erode(image1_mask, image1_mask, Mat(), Point(-1, -1), 5);
    dilate(image1_mask, image1_mask, Mat(), Point(-1, -1), 5);

    vector<vector<Point> > contours;
    findContours(image1_mask, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<vector<Point> > hulls;
    for (int i = 0; i < contours.size(); ++i)
    {
        vector<Point> hull;
        convexHull(contours[i], hull);
        hulls.push_back(hull);
    }

    drawContours(image1_mask, hulls, -1, 255, -1);

    // convex hull
    // teken filled hulls

    // EZ-merge
    Mat image1_bgr_merge_p2[] = {b & image1_mask, g & image1_mask, r & image1_mask};
    Mat image1_skin_p2;
    merge(image1_bgr_merge_p2, 3, image1_skin_p2);
    //cout << image1_skin << endl;

    namedWindow("Display Image 1 (Skin Segmentation) part 2", WINDOW_AUTOSIZE);
    imshow("Display Image 1 (Skin Segmentation) part 2", image1_skin_p2);



    // Image 2 processing
    // ------------------

    // Histograms Equalization all in one.
    Mat image2_equalized;
    equalizeHist(image2, image2_equalized);

    namedWindow("Display Image 2 (Text Segmentation) EQ", WINDOW_AUTOSIZE);
    imshow("Display Image 2 (Text Segmentation) EQ", image2_equalized);

    // Threshold on Histograms Equalization
    Mat image2_th_hq = Mat::zeros(image2.rows, image2.cols, CV_8UC1);
    threshold(image2_equalized, image2_th_hq, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    namedWindow("Display Image 2 (Text Segmentation) TH on EQ", WINDOW_AUTOSIZE);
    imshow("Display Image 2 (Text Segmentation) TH on EQ", image2_th_hq);

    // CLAHE
    Mat image2_CLAHE;
    Ptr<CLAHE> ptr = createCLAHE();
    ptr->setTilesGridSize(Size(15, 15));
    ptr->setClipLimit(1);
    ptr->apply(image2, image2_CLAHE);

    namedWindow("Display Image 2 (Text Segmentation) CLAHE", WINDOW_AUTOSIZE);
    imshow("Display Image 2 (Text Segmentation) CLAHE", image2_CLAHE);

    // Threshold on Histograms Equalization
    Mat image2_th_clahe = Mat::zeros(image2.rows, image2.cols, CV_8UC1);
    threshold(image2_CLAHE, image2_th_clahe, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    namedWindow("Display Image 2 (Text Segmentation) TH on CLAHE", WINDOW_AUTOSIZE);
    imshow("Display Image 2 (Text Segmentation) TH on CLAHE", image2_th_clahe);

    // Sleep & do event loop.
    waitKey(0);

    return 0;
}
