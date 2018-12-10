/**
 * @author Dries Kennes (R0486630)
 */

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//int FPS = 30;
bool SLEEP = false;

void mouseHandler(int event, int x, int y, int, void*)
{
    switch (event)
    {
        case EVENT_LBUTTONUP:
            SLEEP = !SLEEP;
    }
}

// Te ingewikkeld, not worth it.
//int get_delay()
//{
//    static double prev_target_0 = 0;
//    static double prev_target_1 = 0;
//    static long prev = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
//    // +1 to account for this logic & OpenCV event loop.
//    long now = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
//    // delta = time since last call of this function in ms.
//    int delta = (int) (now - prev);
//    prev = now;
//    // target = delay we want to get to target FPS
//    double target = (1000.0/FPS) - delta;
//    // smooth out this with the last by averaging, so we don't end up with oscillations
//    double delay = (target + prev_target_0 + prev_target_1) / 3.0;
//    prev_target_1 = prev_target_0;
//    prev_target_0 = target;
//
//    //cout << "[FPS] Delta: " << delta << " Target:" << target << " Smoothed: " << delay << endl;
//    cout << "[FPS] " << (1000.0/delta) << "\t" << target << "\t" << delta << endl;
//    // Make sure not to return 0.
//    return delay >= 0 ? (int) delay : 1;
//}

int main(int argc, const char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ help h usage ?   |     | Show this massage. }"
                             "{ @<video>         |     | Video filename. }"
    );
    parser.about("Parameters surrounded by <> are required.\n");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string videoFilename(parser.get<string>("@<video>"));
    // Required arguments check
    if (videoFilename.empty())
    {
        cerr << "Missing arguments." << endl;
        parser.printMessage();
        return -1;
    }

    VideoCapture video(videoFilename);
    if (!video.isOpened())
    {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    Scalar color(200, 15, 200);

    namedWindow("Frame");
    setMouseCallback("Frame", mouseHandler);
//    createTrackbar("FPS", "Frame", &FPS, 60);
    cout << "Click to pause & unpause";

    vector<Point> path;

    long start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();

    do
    {
        Mat frame;
        video.read(frame);

        if (frame.empty())
        {
            path.clear();
            long end = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
            cout << "Looping video, played in " << (end-start)/1000.0 << "s" << endl;
            video.set(CAP_PROP_POS_FRAMES, 0);

            cout << "Press the any key to continue" << endl;
            waitKey(0);

            start = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
            continue;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
//        equalizeHist(gray, gray);

        vector<Rect> found, found_filtered;
//        std::vector<double> weights;
        hog.detectMultiScale(gray, found);
//        hog.detectMultiScale(gray, found, weights);

        for (int i = path.size() - 2; i >= 0; i--)
        {
            line(frame, path[i], path[i+1], color);
        }

        if (!found.empty())
        {
            int i = 0;
            const auto &r = found[i];
//            const auto &n = weights[i];
            Point center(r.x + r.width/2, r.y + r.height/2);
            rectangle(frame, r, color);
//            putText(frame, to_string(n), r.br(), FONT_HERSHEY_PLAIN, 1, color);

            if (!path.empty()) line(frame, center, path[path.size()-1], color);
            path.push_back(center);
        }
        imshow("Frame", frame);

        while (SLEEP) if (waitKey(1) == '\x1B') return 0;
    }
    while (waitKey(1) != '\x1B'); // Sleep & do event loop.
    //while (waitKey(get_delay()) != '\x1B'); // Sleep & do event loop.

    return 0;
}
