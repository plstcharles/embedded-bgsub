#pragma once

#include "VibeBGSUtils.hpp"

#include <opencv2/core.hpp>

namespace sky360 {

    class VibeBGS {
    public:
        /// defines the default value for BackgroundSubtractorViBe::m_nColorDistThreshold
        static const size_t DEFAULT_COLOR_DIST_THRESHOLD{20};
        /// defines the default value for BackgroundSubtractorViBe::m_nBGSamples
        static const size_t DEFAULT_NB_BG_SAMPLES{16};
        /// defines the default value for BackgroundSubtractorViBe::m_nRequiredBGSamples
        static const size_t DEFAULT_REQUIRED_NB_BG_SAMPLES{2};
        /// defines the default value for the learning rate passed to BackgroundSubtractorViBe::apply (the 'subsampling' factor in the original ViBe paper)
        static const size_t DEFAULT_LEARNING_RATE{8};

        VibeBGS(size_t nColorDistThreshold = DEFAULT_COLOR_DIST_THRESHOLD,
                size_t nBGSamples = DEFAULT_NB_BG_SAMPLES,
                size_t nRequiredBGSamples = DEFAULT_REQUIRED_NB_BG_SAMPLES,
                size_t learningRate = DEFAULT_LEARNING_RATE);

        void initialize(const Img& _initImg);
        void initialize(const cv::Mat& oInitImg);

        void initializeParallel(const Img& _initImg, int _numProcesses);

        void apply(const Img& _image, Img& _fgmask);
        void apply(const cv::Mat& _image, cv::Mat& _fgmask);

        void applyParallel(const Img& _image, Img& _fgmask);

    private:
        Params m_params;

        std::vector<std::shared_ptr<Img>> m_bgImgSamples;
        std::vector<std::vector<std::shared_ptr<Img>>> m_bgImgSamplesParallel;

        void initialize(const Img& _initImg, std::vector<std::shared_ptr<Img>>& _bgImgSamples);
        static void applyCmp(const Img& image, std::vector<std::shared_ptr<Img>>& bgImg, Img& fgmask, const Params& _params);
    };
}