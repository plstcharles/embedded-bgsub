#include "WeightedMovingVariance.hpp"

// opencv legacy includes
#include <opencv2/imgproc/types_c.h>
#include <execution>
#include <iostream>

using namespace bgslibrary::algorithms;

WeightedMovingVariance::WeightedMovingVariance() 
    : m_params(true, true, 15),
    m_numProcessesParallel(12)
{
    for (int i = 0; i < m_numProcessesParallel; ++i) {
        m_processSeq.push_back(i);
    }
    img_input_prev_parallel.resize(m_numProcessesParallel);
}

WeightedMovingVariance::~WeightedMovingVariance()
{
}

void WeightedMovingVariance::process(const cv::Mat &_imgInput, cv::Mat &_imgOutput)
{
    process(_imgInput, _imgOutput, img_input_prev, m_params);
}

void WeightedMovingVariance::processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput) {
    std::for_each(
        std::execution::par,
        m_processSeq.begin(),
        m_processSeq.end(),
        [&](int np)
        {
            int height = _imgInput.size().height / m_numProcessesParallel;
            int pixelPos = np * _imgInput.size().width * height;
            //std::cout << "np: " << np << ", pos: " << pixelPos << ", height: " << height << std::endl; 
            cv::Mat imgSplit(height, _imgInput.size().width, _imgInput.type(), _imgInput.data + (pixelPos * 3));
            cv::Mat maskPartial(height, _imgInput.size().width, _imgOutput.type(), _imgOutput.data + pixelPos);
            process(imgSplit, maskPartial, img_input_prev_parallel[np], m_params);
        });
}

void WeightedMovingVariance::process(const cv::Mat &img_input, cv::Mat &img_output, std::array<cv::Mat, 2>& img_input_prev, const WeightedMovingVarianceParams& _params)
{
    if (img_input_prev[0].empty())
    {
        img_input.copyTo(img_input_prev[0]);
        return;
    }

    if (img_input_prev[1].empty())
    {
        img_input_prev[0].copyTo(img_input_prev[1]);
        img_input.copyTo(img_input_prev[0]);
        return;
    }

    cv::Mat img_input_f;
    img_input.convertTo(img_input_f, CV_32F, 1. / 255.);

    cv::Mat img_input_prev_1_f;
    img_input_prev[0].convertTo(img_input_prev_1_f, CV_32F, 1. / 255.);

    cv::Mat img_input_prev_2_f;
    img_input_prev[1].convertTo(img_input_prev_2_f, CV_32F, 1. / 255.);

    // Weighted mean
    cv::Mat img_mean_f;

    if (_params.enableWeight)
        img_mean_f = ((img_input_f * 0.5) + (img_input_prev_1_f * 0.3) + (img_input_prev_2_f * 0.2));
    else
        img_mean_f = ((img_input_f * 0.3) + (img_input_prev_1_f * 0.3) + (img_input_prev_2_f * 0.3));

    // Weighted variance
    cv::Mat img_1_f;
    cv::Mat img_2_f;
    cv::Mat img_3_f;
    cv::Mat img_4_f;

    if (_params.enableWeight)
    {
        computeWeightedVariance(img_input_f, img_mean_f, 0.5, img_1_f);
        computeWeightedVariance(img_input_prev_1_f, img_mean_f, 0.3, img_2_f);
        computeWeightedVariance(img_input_prev_2_f, img_mean_f, 0.2, img_3_f);
        img_4_f = (img_1_f + img_2_f + img_3_f);
    }
    else
    {
        computeWeightedVariance(img_input_f, img_mean_f, 0.3, img_1_f);
        computeWeightedVariance(img_input_prev_1_f, img_mean_f, 0.3, img_2_f);
        computeWeightedVariance(img_input_prev_2_f, img_mean_f, 0.3, img_3_f);
        img_4_f = (img_1_f + img_2_f + img_3_f);
    }

    // Standard deviation
    cv::Mat img_sqrt_f(img_input.size(), CV_32F);
    cv::sqrt(img_4_f, img_sqrt_f);
    cv::Mat img_sqrt(img_input.size(), CV_8U);
    double minVal, maxVal;
    minVal = 0.;
    maxVal = 1.;
    img_sqrt_f.convertTo(img_sqrt, CV_8U, 255.0 / (maxVal - minVal), -minVal);

    if (img_sqrt.channels() == 3)
        cv::cvtColor(img_sqrt, img_sqrt, CV_BGR2GRAY);

    if (_params.enableThreshold)
        cv::threshold(img_sqrt, img_sqrt, _params.threshold, 255, cv::THRESH_BINARY);

    memcpy(img_output.data, img_sqrt.data, img_output.size().width * img_output.size().height);

    img_input_prev[0].copyTo(img_input_prev[1]);
    img_input.copyTo(img_input_prev[0]);
}

void WeightedMovingVariance::computeWeightedVariance(const cv::Mat &img_input_f, const cv::Mat &img_mean_f, const double weight, cv::Mat& img_f)
{
    // ERROR in return (weight * ((cv::abs(img_input_f - img_mean_f))^2.));

    cv::Mat img_f_absdiff(img_input_f.size(), CV_32F);
    cv::absdiff(img_input_f, img_mean_f, img_f_absdiff);
    cv::Mat img_f_pow(img_input_f.size(), CV_32F);
    cv::pow(img_f_absdiff, 2.0, img_f_pow);
    img_f = weight * img_f_pow;
}
