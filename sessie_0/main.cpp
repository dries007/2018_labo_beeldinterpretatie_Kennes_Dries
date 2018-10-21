/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, const char **argv)
{
    // @ = positional (enforced by OpenCL)
    // <> = required (enforced by application)
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<file1>         |     | File 1 to display (Greyscale). }"
                             "{ @<file2>         |     | File 2 to display (Color). }"
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
    Mat image1 = imread(file1, 1); // Greyscale
    Mat image2 = imread(file2, 1); // Color
    if (!image1.data)
    {
        cout << "Could not load file 1." << endl;
        return -1;
    }
    if (!image2.data)
    {
        cout << "Could not load file 2." << endl;
        return -1;
    }

    // Display images as loaded.
    namedWindow("Display Image 1 (Greyscale)", WINDOW_AUTOSIZE);
    namedWindow("Display Image 2 (Color)", WINDOW_AUTOSIZE);
    imshow("Display Image 1 (Greyscale)", image1);
    imshow("Display Image 2 (Color)", image2);

    // Split color image into RGB
    Mat image2_bgr[3];
    split(image2, image2_bgr);

    namedWindow("Display Image 2 R", WINDOW_AUTOSIZE);
    namedWindow("Display Image 2 G", WINDOW_AUTOSIZE);
    namedWindow("Display Image 2 B", WINDOW_AUTOSIZE);
    imshow("Display Image 2 R", image2_bgr[2]);
    imshow("Display Image 2 G", image2_bgr[1]);
    imshow("Display Image 2 B", image2_bgr[0]);

    // Greyify color image
    Mat image2_grey;
    cvtColor(image2, image2_grey, COLOR_BGR2GRAY);

    namedWindow("Display Image 2 Grey", WINDOW_AUTOSIZE);
    imshow("Display Image 2 Grey", image2_grey);

    // Loop all pixels
    // Roughly equivalent: cout << image2_grey << endl;
    for (int r = 0; r < image2.rows; ++r)
    {
        for (int c = 0; c < image2.cols; ++c)
        {
            int v = image2.at<uchar>(r, c);
            cout << v << ", ";
        }
        cout << endl;
    }
    cout << endl;


    // Draw canvas with some stuff on it.
    Mat canvas = Mat::zeros(255, 255, CV_8UC3);
    rectangle(canvas, Rect(120, 80, 25, 60), Scalar(255, 175, 0), 1);
    circle(canvas, Point(100, 50), 50, Scalar(0, 175, 255), 2);
    namedWindow("Display Canvas", WINDOW_AUTOSIZE);
    imshow("Display Canvas", canvas);

    // Sleep & do event loop.
    waitKey(0);

    return 0;
}
