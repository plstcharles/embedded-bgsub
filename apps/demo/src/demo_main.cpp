#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "api.hpp"
#include "profiling.hpp"

const char* keys =
{
    "{help h | | show help message}{@camera_number| 0 | camera number}"
};

static void help(const char** argv)
{
    std::cout << "\nThis is a demo to test Background Subtracting\n"
        "This reads from video camera (0 by default, or the camera number the user enters)\n";
        "Usage: \n\t";
    std::cout << argv[0] << " [camera number]\n";
}

int main(int argc, const char** argv) {
    int learningRate = 10;
    cv::VideoCapture cap;
    BackgroundSubtractorViBe_3ch vibe;
    cv::CommandLineParser parser(argc, argv, keys);

    double freq = initFrequency();
    std::cout << "Current frequency: " << freq << "\n";

    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    int camNum = parser.get<int>(0);
    cap.open(camNum);
    if (!cap.isOpened())
    {
        help(argv);
        std::cout << "***Could not initialize capturing...***\n";
        std::cout << "Current parameter's value: \n";
        parser.printMessage();
        return -1;
    }

    double frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << "Capture size: " << (int)frameWidth << " x " << (int)frameHeight << "\n";

    cv::namedWindow("ViBe Demo", 0);

    cv::Mat frame, vibeMask;
    bool vibeInit = true;
    long numFrames = 0;
    double totalTime = 0;

    while (true) {
        cap >> frame;
        //cv::cvtColor(frame, hsv, COLOR_BGR2GRAY);
        if (frame.empty()) {
            break;
        }
        if (vibeInit)
        {
            vibe.initialize(frame);
            vibeInit = false;
            frame.copyTo(vibeMask);
        } else {
            double startTime = getAbsoluteTime();
            vibe.apply(frame, vibeMask, learningRate);
            double endTime = getAbsoluteTime();
            totalTime += endTime - startTime;
            ++numFrames;

            if (numFrames % 100 == 0) {
                std::cout << "Framerate: " << (numFrames / totalTime) << "fps\n";
            }
        }
        cv::imshow("CamShift Demo", vibeMask);

        char c = (char)cv::waitKey(10);
        if (c == 27) {
            break;
        }
    }

    return 0;
}
