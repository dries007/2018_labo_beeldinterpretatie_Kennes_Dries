/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void run(int pos, void *userdata);

Mat input_image;
Mat object_image;

int detector_int = 0;

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<object>        |     | Object image. }"
                             "{ @<file>          |     | Input image. }"
    );
    parser.about("Parameters surrounded by <> are required.\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string object_name(parser.get<string>("@<object>"));
    string file_name(parser.get<string>("@<file>"));
    // Required arguments check
    if (object_name.empty() || file_name.empty())
    {
        cout << "Missing arguments." << endl;
        parser.printMessage();
        return -1;
    }

    // Load images & error check
    object_image = imread(object_name, IMREAD_COLOR);
    input_image = imread(file_name, IMREAD_COLOR);

    if (!input_image.data || !object_image.data)
    {
        cout << "Could not load files." << endl;
        return -1;
    }

    // Display image as loaded.
    namedWindow("Object Image", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    imshow("Object Image", object_image);
    namedWindow("Input Image", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    imshow("Input Image", input_image);

    namedWindow("Output", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    createTrackbar("Method: \n 0: ORB\n 1: BRISK \n 2: AKAZE", "Output", &detector_int, 2, run);

    run(0, NULL);

    // Sleep & do event loop.
    waitKey(0);

    return 0;
}

void run(int pos, void *userdata)
{
    Ptr<FeatureDetector> detector;

    switch (detector_int)
    {
        default:
        case 0: detector = ORB::create(); break;
        case 1: detector = BRISK::create(); break;
        case 2: detector = AKAZE::create(); break;
    }

    vector<KeyPoint> object_keypoints;
    vector<KeyPoint> image_keypoints;

    Mat object_descriptors;
    Mat image_descriptors;

    detector->detect(object_image, object_keypoints);
    detector->compute(object_image, object_keypoints, object_descriptors);
    detector->detect(input_image, image_keypoints, image_descriptors);
    detector->compute(input_image, image_keypoints, image_descriptors);

    Mat object_image_keypoints;
    Mat input_image_keypoints;

    drawKeypoints(object_image, object_keypoints, object_image_keypoints);
    drawKeypoints(input_image, image_keypoints, input_image_keypoints);

    imshow("Object Image", object_image_keypoints);
    imshow("Input Image", input_image_keypoints);

    Ptr<DescriptorMatcher> matcher = BFMatcher::create();
    vector<DMatch> matches;
    matcher->match(object_descriptors, image_descriptors, matches);

    cout << "Obj keyp: " << object_keypoints.size() << " Img keyp: " << image_keypoints.size() << " Matches: " << matches.size() << endl;

    Mat matches_image;
    // Order of arguments matters. It must be the same as the matcher.
    drawMatches(object_image, object_keypoints, input_image, image_keypoints, matches, matches_image);

    namedWindow("Matches", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    imshow("Matches", matches_image);

    // https://docs.opencv.org/3.0-beta/doc/tutorials/features2d/feature_homography/feature_homography.html
    double max_dist = 0; double min_dist = 100;
    for(int i = 0; i < object_descriptors.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist == 0) {
            cout << "Exact match " << i << endl;
            continue;
        }
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout << "Max dist: " << max_dist << " Min dis: " << min_dist << endl;
    std::vector< DMatch > good_matches;
    for(int i = 0; i < object_descriptors.rows; i++)
    {
        if (matches[i].distance < 3*min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }
    cout << "Good matches: " << good_matches.size() << endl;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (int i = 0; i < good_matches.size(); i ++)
    {
        obj.push_back(object_keypoints[matches[i].queryIdx].pt);
        scene.push_back(image_keypoints[matches[i].trainIdx].pt);
    }

    Mat h = findHomography(obj, scene, RANSAC);

    Mat output;

    drawMatches(object_image, object_keypoints, input_image, image_keypoints, good_matches, output);

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint(object_image.cols, 0);
    obj_corners[2] = cvPoint(object_image.cols, object_image.rows); obj_corners[3] = cvPoint(0, object_image.rows);
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform(obj_corners, scene_corners, h);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(output, scene_corners[0] + Point2f(object_image.cols, 0), scene_corners[1] + Point2f(object_image.cols, 0), Scalar(0, 255, 0), 4);
    line(output, scene_corners[1] + Point2f(object_image.cols, 0), scene_corners[2] + Point2f(object_image.cols, 0), Scalar( 0, 255, 0), 4);
    line(output, scene_corners[2] + Point2f(object_image.cols, 0), scene_corners[3] + Point2f(object_image.cols, 0), Scalar( 0, 255, 0), 4);
    line(output, scene_corners[3] + Point2f(object_image.cols, 0), scene_corners[0] + Point2f(object_image.cols, 0), Scalar( 0, 255, 0), 4);

    imshow("Output", output);
}
