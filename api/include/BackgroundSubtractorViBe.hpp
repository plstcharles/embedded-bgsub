
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// @@@@@@@@
//
// Video Background Extractor (ViBe); originally proposed by O. Barnich and M. Van Droogenbroeck.
//
// CAUTION: this implementation of ViBe is very naive, and not optimized at all. It was used as a
// code sandbox for early versions of LOBSTER. If you want a well-implemented, fully vectorized
// version for testing/evaluation, contact the original authors via http://www.vibeinmotion.com/
//
// Note that ViBe is patented in the US, Europe and Japan; this implementation is offered for
// testing purposes only. For commercial use, refer to the original author's licensing guide on
// their website: http://www.vibeinmotion.com/Licensing.aspx
//
// For information on the patent, see http://www.google.com/patents/WO2009007198A1
//
// Journal paper: O. Barnich and M. Van Droogenbroeck. "ViBe: A universal background subtraction
// algorithm for video sequences" (IEEE Trans. Image Processing 2011)
//
// @@@@@@@@

#include <opencv2/video/background_segm.hpp>
#include "pcg32.hpp"

/// defines the default value for BackgroundSubtractorViBe::m_nColorDistThreshold
#define BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD (20)
/// defines the default value for BackgroundSubtractorViBe::m_nBGSamples
#define BGSVIBE_DEFAULT_NB_BG_SAMPLES (20)
/// defines the default value for BackgroundSubtractorViBe::m_nRequiredBGSamples
#define BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
/// defines the default value for the learning rate passed to BackgroundSubtractorViBe::apply (the 'subsampling' factor in the original ViBe paper)
#define BGSVIBE_DEFAULT_LEARNING_RATE (16)
/// defines the internal threshold adjustment factor to use when determining if the variation of a single channel is enough to declare the pixel as foreground
#define BGSVIBE_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.60f)
/// defines whether we should use single channel variation checks for fg/bg segmentation validation or not
#define BGSVIBE_USE_SC_THRS_VALIDATION 0
/// defines whether we should use L1 distance or L2 distance for change detection
#define BGSVIBE_USE_L1_DISTANCE_CHECK 0

/// ViBe foreground-background segmentation algorithm (abstract version)
class BackgroundSubtractorViBe : public cv::BackgroundSubtractor {
public:
    /// full constructor
    BackgroundSubtractorViBe(size_t nColorDistThreshold = BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
        size_t nBGSamples = BGSVIBE_DEFAULT_NB_BG_SAMPLES,
        size_t nRequiredBGSamples = BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    /// default destructor
    virtual ~BackgroundSubtractorViBe();
    /// (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg) = 0;
    /// primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = BGSVIBE_DEFAULT_LEARNING_RATE) = 0;
    /// returns a copy of the latest reconstructed background image
    void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
    /// number of different samples per pixel/block to be taken from input frames to build the background model ('N' in the original ViBe paper)
    const size_t m_nBGSamples;
    /// number of similar samples needed to consider the current pixel/block as 'background' ('#_min' in the original ViBe paper)
    const size_t m_nRequiredBGSamples;
    /// background model pixel intensity samples
    std::vector<cv::Mat> m_voBGImg;
    /// input image size
    cv::Size m_oImgSize;
    /// absolute color distance threshold ('R' or 'radius' in the original ViBe paper)
    const size_t m_nColorDistThreshold;
    /// defines whether or not the subtractor is fully initialized
    bool m_bInitialized;

    // Second version, not doing square root
    static inline size_t L2dist3Squared(const cv::Vec<uchar, 3>& a, const cv::Vec<uchar, 3>& b) {
        const long r0 = a[0] - b[0];
        const long r1 = a[1] - b[1];
        const long r2 = a[2] - b[2];
        return (r0 * r0) + (r1 * r1) + (r2 * r2);
    }

    	/// returns the neighbor location for the specified random index & original pixel location; also guards against out-of-bounds values via image/border size check
	static inline void getNeighborPosition_3x3(int& nNeighborCoord_X, int& nNeighborCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const cv::Size& oImageSize) {
		typedef std::array<int, 2> Nb;
		static const std::array<std::array<int, 2>, 8> s_anNeighborPattern = {
				Nb{-1, 1},Nb{0, 1},Nb{1, 1},
				Nb{-1, 0},         Nb{1, 0},
				Nb{-1,-1},Nb{0,-1},Nb{1,-1},
		};
		const int r = Pcg32::fast() % 8;
		nNeighborCoord_X = nOrigCoord_X + s_anNeighborPattern[r][0];
		nNeighborCoord_Y = nOrigCoord_Y + s_anNeighborPattern[r][1];

        if (nNeighborCoord_X < 0)
			nNeighborCoord_X = 0;
		else if (nNeighborCoord_X >= oImageSize.width)
			nNeighborCoord_X = oImageSize.width - 1;
		if (nNeighborCoord_Y < 0)
			nNeighborCoord_Y = 0;
		else if (nNeighborCoord_Y >= oImageSize.height)
			nNeighborCoord_Y = oImageSize.height - 1;
	}
};

/// ViBe foreground-background segmentation algorithm (1ch/grayscale version)
class BackgroundSubtractorViBe_1ch : public BackgroundSubtractorViBe {
public:
    /// full constructor
    BackgroundSubtractorViBe_1ch(size_t nColorDistThreshold = BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
        size_t nBGSamples = BGSVIBE_DEFAULT_NB_BG_SAMPLES,
        size_t nRequiredBGSamples = BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    /// default destructor
    virtual ~BackgroundSubtractorViBe_1ch();
    /// (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg);
    /// primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = BGSVIBE_DEFAULT_LEARNING_RATE);
};

/// ViBe foreground-background segmentation algorithm (3ch/RGB version)
class BackgroundSubtractorViBe_3ch : public BackgroundSubtractorViBe {
public:
    /// full constructor
    BackgroundSubtractorViBe_3ch(size_t nColorDistThreshold = BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
        size_t nBGSamples = BGSVIBE_DEFAULT_NB_BG_SAMPLES,
        size_t nRequiredBGSamples = BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    /// default destructor
    virtual ~BackgroundSubtractorViBe_3ch();
    /// (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg);
    /// primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(const cv::InputArray image, cv::OutputArray fgmask, double learningRate = BGSVIBE_DEFAULT_LEARNING_RATE);

    void initializeParallel(const cv::Mat& oInitImg, int numProcesses);
    void applyParallel(const cv::InputArray image, cv::OutputArray fgmask, double learningRate = BGSVIBE_DEFAULT_LEARNING_RATE);

private:
    const size_t m_nColorDistThresholdSquared;

    int m_numProcessesParallel;
    std::vector<std::vector<cv::Mat>> m_voBGImgParallel;
    std::vector<int> m_processSeq;
    std::vector<cv::Rect> m_rectImgs;
    std::vector<cv::Mat> m_outSplit;

    void splitImages(const cv::Mat& inputImg, std::vector<cv::Mat>& outputImages, int numSlices);
    void joinImages(std::vector<cv::Mat>& outputImages, cv::Mat& outputImg);
    void applyCmp(const cv::Mat& _image, std::vector<cv::Mat>& _bg, cv::Mat& _fgmask, double learningRate);
};
