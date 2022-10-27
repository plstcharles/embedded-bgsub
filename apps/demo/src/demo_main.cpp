#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

//#include "BackgroundSubtractorViBe.hpp"
#include "VibeBGS.hpp"
#include "WeightedMovingVariance.hpp"
#include "profiling.hpp"

int main(int argc, const char** argv) {
    cv::VideoCapture cap;
    bgslibrary::algorithms::WeightedMovingVariance wmv;
    sky360::VibeBGS vibeBGS;

    double freq = initFrequency();

    cv::CommandLineParser parser(argc, argv, keys);

    int camNum = parser.get<int>(0);
    cap.open(camNum);
    //cap.open("E:\\source\\sky360\\embedded-bgsub\\Dahua-20220901-184734.mp4");
    if (!cap.isOpened())
    {
        std::cout << "***Could not initialize capturing...***\n";
        std::cout << "Current parameter's value: \n";
        parser.printMessage();
        return -1;
    }

    double frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Capture size: " << (int)frameWidth << " x " << (int)frameHeight << std::endl;

    cv::namedWindow("BGS Demo", 0);

    cv::Mat frame, greyFrame;
    long numFrames = 0;
    double totalTime = 0;

    cap >> frame;
    if (frame.type() != CV_8UC3) {
        std::cout << "Image type not supported" << std::endl;
        return -1;
    }

    cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
    vibeBGS.initialize(greyFrame, 12);
    std::cout << "initializeParallel" << std::endl;

    cv::Mat bgsMask(frame.size(), CV_8UC1);

    cv::imshow("BGS Demo", frame);

    std::cout << "Enter loop" << std::endl;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "No image" << std::endl;
            break;
        }
        cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);

        double startTime = getAbsoluteTime();
        //wmv.process(frame, bgsMask);
        //wmv.processParallel(frame, bgsMask);
        vibeBGS.apply(greyFrame, bgsMask);
        double endTime = getAbsoluteTime();
        totalTime += endTime - startTime;
        ++numFrames;
        //std::cout << "Frame: " << numFrames << std::endl;

        if (numFrames % 100 == 0) {
            std::cout << "Framerate: " << (numFrames / totalTime) << " fps" << std::endl;
        }
        cv::imshow("BGS Demo", bgsMask);

        char c = (char)cv::waitKey(10);
        if (c == 27) {
            std::cout << "Escape key pressed" << std::endl;
            break;
        }
    }
    std::cout << "Exit loop\n" << std::endl;

    return 0;
}
