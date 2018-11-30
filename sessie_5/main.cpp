/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void drawpoints();
void mouseHandler(int event, int x, int y, int, void*);

//C++, award winner for ugliest C-like syntax ever.
class PointMouse: public Point {
public:
    PointMouse(int x, int y, bool positive) : Point(x, y), positive(positive) { };
    bool positive;
};

Mat input_image;
Mat hsv;
vector<PointMouse> points;

void show_masked(const String& name, const Mat& results)
{
    Mat mask = Mat::zeros(hsv.size(), CV_8UC1);
    for (int r = 0; r < hsv.rows; r++)
    {
        for (int c = 0; c < hsv.cols; c++)
        {
            if (results.at<float>(r * hsv.cols + c) != 0)
            {
                mask.at<uchar>(r, c) = 1;
            }
        }
    }

    // Clear
    erode(mask, mask, Mat(), Point(-1, -1), 2);
    dilate(mask, mask, Mat(), Point(-1, -1), 2);

    // Glue
    dilate(mask, mask, Mat(), Point(-1, -1), 5);
    erode(mask, mask, Mat(), Point(-1, -1), 5);

    Mat masked;
    input_image.copyTo(masked, mask);
    imshow(name, masked);
}

void knn(const Mat& training, const Mat& labels, const Mat& samples)
{
    Ptr<ml::KNearest> knn = ml::KNearest::create();

    knn->setIsClassifier(true);
    knn->setAlgorithmType(ml::KNearest::Types::BRUTE_FORCE);
    knn->setDefaultK(3);

    knn->train(training, ml::ROW_SAMPLE, labels);

    Mat results;
    knn->findNearest(samples, knn->getDefaultK(), results);

    show_masked("KNN Masked", results);
}

void nbc(const Mat& training, const Mat& labels, const Mat& samples)
{
    Ptr<ml::NormalBayesClassifier> nbc = ml::NormalBayesClassifier::create();

    nbc->train(training, ml::ROW_SAMPLE, labels);
    Mat results;
    nbc->predict(samples, results);

    show_masked("NBC Masked", results);
}

void svm(const Mat& training, const Mat& labels, const Mat& samples)
{
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(training, ml::ROW_SAMPLE, labels);
    Mat results;
    svm->predict(samples, results);

    show_masked("SVM Masked", results);
}

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<object>        |     | Object image. }"
    );
    parser.about("Parameters surrounded by <> are required.\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string object_name(parser.get<string>("@<object>"));
    // Required arguments check
    if (object_name.empty())
    {
        cout << "Missing arguments." << endl;
        parser.printMessage();
        return -1;
    }

    // Load images & error check
    input_image = imread(object_name, IMREAD_COLOR);

    if (!input_image.data)
    {
        cout << "Could not load files." << endl;
        return -1;
    }

    // Display image as loaded.
    namedWindow("Input Image");
    //imshow("Input Image", input_image);

    setMouseCallback("Input Image", mouseHandler);

    cout << "LEFT CLICK: Add GOOD points." << endl;
    cout << "RIGHT CLICK: Add BAD points." << endl;
    cout << "MIDDLE CLICK: Erase last point." << endl;
    cout << "Hit enter to run the algorithm." << endl;

    { // Load points
        ifstream f(object_name + ".txt");
        if (f)
        {
            string str;
            while (getline(f, str))
            {
                istringstream iss(str);
                int x, y;
                char pos;
                iss >> x >> y >> pos;
                points.push_back(PointMouse(x, y, pos == 'P'));
            }
        }
    }

    // RGB (BGR) -> HSV
    hsv = Mat(input_image.clone());
    // Remove green (BGR) -> 1e col
    // https://stackoverflow.com/q/23510571
    for (int i = 0; i < input_image.rows; i++) hsv.row(i).reshape(1, input_image.cols).col(1).setTo(Scalar(0));
    cvtColor(hsv, hsv, COLOR_BGR2HSV);

    // Blur a little
    GaussianBlur(hsv, hsv, Size(5, 5), 0);

    // Loop so adding points on the go is possible.
    do
    {
        drawpoints();

        { // Save points
            ofstream f(object_name + ".txt");
            for (PointMouse point : points)
            {
                f << point.x << ' ' << point.y << ' ' << (point.positive ? 'P' : 'N') << '\n';
            }
        }

        // Everything below is meaningless (and breaks) if there are no points, so stop here.
        if (points.size() == 0) continue;

        // Training(Rows: 1 per point, Cols: HSV)
        // Labels(Rows: 1 per point, Cols: Classifier (1 = yes, 0 = no))
        Mat training(points.size(), 3, CV_32FC1);
        Mat labels(points.size(), 1, CV_32SC1);

        // Training data & labels.
        for (int i = 0; i < points.size(); i++)
        {
            Vec3b desc = hsv.at<Vec3b>(points[i]);
            training.at<float>(i, 0) = desc[0];
            training.at<float>(i, 1) = desc[1];
            training.at<float>(i, 2) = desc[2];
            labels.at<int>(i, 0) = points[i].positive ? 1 : 0;
        }

        // You can't seem to do this any better way.
        Mat samples(hsv.rows * hsv.cols, 3, CV_32FC1);
        for (int r = 0; r < hsv.rows; r++)
        {
            for (int c = 0; c < hsv.cols; c++)
            {
                int i = r * hsv.cols + c;
                Vec3b desc = hsv.at<Vec3b>(r, c);
                samples.at<float>(i, 0) = desc[0];
                samples.at<float>(i, 1) = desc[1];
                samples.at<float>(i, 2) = desc[2];
            }
        }

        knn(training, labels, samples);
        nbc(training, labels, samples);
        svm(training, labels, samples);
    }
    while (waitKey(0) != '\x1B'); // Sleep & do event loop.

    return 0;
}

bool isPointValid(int x, int y)
{
    return x >= 0 && y >= 0 && x < input_image.cols && y < input_image.rows;
}

void mouseHandler(int event, int x, int y, int, void*)
{
    switch (event)
    {
        case EVENT_LBUTTONUP:
            if (!isPointValid(x, y)) return;
            cout << "Mouse event: L UP " << x << ";" << y << endl;
            points.push_back(PointMouse(x, y, true));
            drawpoints();
            break;

        case EVENT_RBUTTONUP:
            if (!isPointValid(x, y)) return;
            cout << "Mouse event: R UP " << x << ";" << y << endl;
            points.push_back(PointMouse(x, y, false));
            drawpoints();
            break;

        case EVENT_MBUTTONUP:
            cout << "Mouse event: M UP " << x << ";" << y << endl;
            if (!points.empty()) points.pop_back();
            drawpoints();
            break;
    }
}

void drawpoints()
{
    cout << "We have " << points.size() << " points." << endl;

    Mat image(input_image.clone());

    for (PointMouse point : points)
    {
        circle(image, point, 2, point.positive ? Scalar(79, 255, 84) : Scalar(76, 76, 255), 1);
    }

    imshow("Input Image", image);
}
