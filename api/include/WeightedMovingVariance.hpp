#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <vector>

namespace bgslibrary
{
    namespace algorithms
    {
        struct WeightedMovingVarianceParams
        {
            WeightedMovingVarianceParams(bool _enableWeight,
                                        bool _enableThreshold,
                                        int _threshold)
                : enableWeight(_enableWeight),
                enableThreshold(_enableThreshold),
                threshold(_threshold)
            {}

            const bool enableWeight;
            const bool enableThreshold;
            const int threshold;
        };

        class WeightedMovingVariance
        {
        public:
            WeightedMovingVariance();
            ~WeightedMovingVariance();

            void process(const cv::Mat &img_input, cv::Mat &img_output);
            void processParallel(const cv::Mat &_imgInput, cv::Mat &_imgOutput);

        private:
            std::array<cv::Mat, 2> img_input_prev;

            const int m_numProcessesParallel;
            std::vector<int> m_processSeq;
            std::vector<std::array<cv::Mat, 2>> img_input_prev_parallel;

            const WeightedMovingVarianceParams m_params;

            static void process(const cv::Mat &img_input, cv::Mat &img_output, std::array<cv::Mat, 2>& img_input_prev, const WeightedMovingVarianceParams& _params);

            static void computeWeightedVariance(const cv::Mat &img_input_f, const cv::Mat &img_mean_f, const double weight, cv::Mat& img_f);
        };
    }
}
