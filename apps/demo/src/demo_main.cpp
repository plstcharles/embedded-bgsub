#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

//#include "BackgroundSubtractorViBe.hpp"
#include "VibeBGS.hpp"
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
    cv::VideoCapture cap;
    //BackgroundSubtractorViBe_3ch vibe;
    sky360::VibeBGS vibeBGS;
    cv::CommandLineParser parser(argc, argv, keys);

    double freq = initFrequency();
    //std::cout << "Current frequency: " << freq << "\n";

    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    int camNum = parser.get<int>(0);
    //cap.open(camNum);
    cap.open("E:\\source\\sky360\\embedded-bgsub\\Dahua-20220901-182310.mp4");
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

    std::cout << "Capture size: " << (int)frameWidth << " x " << (int)frameHeight << std::endl;

    cv::namedWindow("ViBe Demo", 0);

    cv::Mat frame;
    long numFrames = 0;
    double totalTime = 0;

    cap >> frame;
    if (frame.type() != CV_8UC3) {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    //vibeBGS.initialize(frame);
    vibeBGS.initializeParallel(frame, 8);
    cv::Mat vibeMask(frame.size(), CV_8UC1);

    cv::imshow("ViBe Demo", frame);

    std::cout << "Enter loop" << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "No image\n";
            break;
        }

        double startTime = getAbsoluteTime();
        //vibeBGS.apply(frame, vibeMask);
        vibeBGS.applyParallel(frame, vibeMask);
        double endTime = getAbsoluteTime();
        totalTime += endTime - startTime;
        ++numFrames;
        //std::cout << "Frame: " << numFrames << std::endl;

        if (numFrames % 100 == 0) {
            std::cout << "Framerate: " << (numFrames / totalTime) << " fps" << std::endl;
        }
        cv::imshow("ViBe Demo", vibeMask);

        char c = (char)cv::waitKey(10);
        if (c == 27) {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
    }
    std::cout << "Exit loop\n" << std::endl;

    return 0;
}
