
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

#include "BackgroundSubtractorViBe.hpp"
#include "vibeUtils.hpp"

#include <execution>

BackgroundSubtractorViBe::BackgroundSubtractorViBe(size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
	m_nBGSamples(nBGSamples),
	m_nRequiredBGSamples(nRequiredBGSamples),
	m_voBGImg(nBGSamples),
	m_nColorDistThreshold(nColorDistThreshold),
	m_bInitialized(false) {}

BackgroundSubtractorViBe::~BackgroundSubtractorViBe() {}

void BackgroundSubtractorViBe::getBackgroundImage(cv::OutputArray backgroundImage) const {
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize, CV_32FC(m_voBGImg[0].channels()));
	for (size_t n = 0; n < m_nBGSamples; ++n) {
		for (int y = 0; y < m_oImgSize.height; ++y) {
			for (int x = 0; x < m_oImgSize.width; ++x) {
				const size_t idx_uchar = m_voBGImg[n].step.p[0] * y + m_voBGImg[n].step.p[1] * x;
				const size_t idx_flt32 = idx_uchar * 4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data + idx_flt32);
				const uchar* const oBGImgPtr = m_voBGImg[n].data + idx_uchar;
				for (size_t c = 0; c < (size_t)m_voBGImg[n].channels(); ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c]) / m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage, CV_8U);
}

BackgroundSubtractorViBe_1ch::BackgroundSubtractorViBe_1ch(size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
	BackgroundSubtractorViBe(nColorDistThreshold, nBGSamples, nRequiredBGSamples) {}

BackgroundSubtractorViBe_1ch::~BackgroundSubtractorViBe_1ch() {}

void BackgroundSubtractorViBe_1ch::initialize(const cv::Mat& oInitImg) {
	m_oImgSize = oInitImg.size();
	for (size_t s = 0; s < m_nBGSamples; s++) {
		m_voBGImg[s].create(m_oImgSize, CV_8UC1);
		m_voBGImg[s] = cv::Scalar(0);
		for (int y_orig = 0; y_orig < m_oImgSize.height; y_orig++) {
			for (int x_orig = 0; x_orig < m_oImgSize.width; x_orig++) {
				int y_sample, x_sample;
				lv::getSamplePosition_7x7_std2(Pcg32::fast(), x_sample, y_sample, x_orig, y_orig, 0, m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig, x_orig) = oInitImg.at<uchar>(y_sample, x_sample);
			}
		}
	}
	m_bInitialized = true;
}

void BackgroundSubtractorViBe_1ch::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	cv::Mat oInputImg = _image.getMat();
	_fgmask.create(m_oImgSize, CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);
	const size_t nLearningRate = (size_t)ceil(learningRate);
	for (int y = 0; y < m_oImgSize.height; y++) {
		for (int x = 0; x < m_oImgSize.width; x++) {
			size_t nGoodSamplesCount = 0, nSampleIdx = 0;
			while (nGoodSamplesCount < m_nRequiredBGSamples && nSampleIdx < m_nBGSamples) {
				if (lv::L1dist(oInputImg.at<uchar>(y, x), m_voBGImg[nSampleIdx].at<uchar>(y, x)) < m_nColorDistThreshold)
					nGoodSamplesCount++;
				nSampleIdx++;
			}
			if (nGoodSamplesCount < m_nRequiredBGSamples)
				oFGMask.at<uchar>(y, x) = UCHAR_MAX;
			else {
				if ((Pcg32::fast() % nLearningRate) == 0)
					m_voBGImg[Pcg32::fast() % m_nBGSamples].at<uchar>(y, x) = oInputImg.at<uchar>(y, x);
				if ((Pcg32::fast() % nLearningRate) == 0) {
					int x_rand, y_rand;
					getNeighborPosition_3x3(x_rand, y_rand, x, y, m_oImgSize);
					m_voBGImg[Pcg32::fast() % m_nBGSamples].at<uchar>(y_rand, x_rand) = oInputImg.at<uchar>(y, x);
				}
			}
		}
	}
}

BackgroundSubtractorViBe_3ch::BackgroundSubtractorViBe_3ch(size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
	BackgroundSubtractorViBe(nColorDistThreshold, nBGSamples, nRequiredBGSamples),
	m_nColorDistThresholdSquared((nColorDistThreshold * 3) * (nColorDistThreshold * 3)) {}

BackgroundSubtractorViBe_3ch::~BackgroundSubtractorViBe_3ch() {}

void BackgroundSubtractorViBe_3ch::initialize(const cv::Mat& oInitImgRGB) {
	m_oImgSize = oInitImgRGB.size();
	int y_sample, x_sample;
	for (size_t s = 0; s < m_nBGSamples; s++) {
		m_voBGImg[s].create(m_oImgSize, CV_8UC3);
		m_voBGImg[s] = cv::Scalar(0, 0, 0);
		for (int y_orig = 0; y_orig < m_oImgSize.height; y_orig++) {
			for (int x_orig = 0; x_orig < m_oImgSize.width; x_orig++) {
				lv::getSamplePosition_7x7_std2(Pcg32::fast(), x_sample, y_sample, x_orig, y_orig, 0, m_oImgSize);
				m_voBGImg[s].at<cv::Vec3b>(y_orig, x_orig) = oInitImgRGB.at<cv::Vec3b>(y_sample, x_sample);
			}
		}
	}
	m_bInitialized = true;
}

void BackgroundSubtractorViBe_3ch::apply(const cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	cv::Mat oInputImgRGB = _image.getMat();
	cv::Mat oFGMask = _fgmask.getMat();

	applyCmp(oInputImgRGB, m_voBGImg, oFGMask, learningRate);

	// oFGMask = cv::Scalar_<uchar>(0);

	// const size_t nLearningRate = (size_t)ceil(learningRate);

	// for (int y = 0; y < m_oImgSize.height; ++y) {
	// 	for (int x = 0; x < m_oImgSize.width; ++x) {
	// 		size_t nGoodSamplesCount = 0, nSampleIdx = 0;
	// 		while (nGoodSamplesCount < m_nRequiredBGSamples && nSampleIdx < m_nBGSamples) {
	// 			const cv::Vec3b& in = oInputImgRGB.at<cv::Vec3b>(y, x);
	// 			const cv::Vec3b& bg = m_voBGImg[nSampleIdx].at<cv::Vec3b>(y, x);
	// 			//if (lv::L2dist(in, bg) < m_nColorDistThreshold * 3)
	// 			if (L2dist3Squared(in, bg) < m_nColorDistThresholdSquared)
	// 				nGoodSamplesCount++;
	// 			nSampleIdx++;
	// 		}
	// 		if (nGoodSamplesCount < m_nRequiredBGSamples)
	// 			oFGMask.at<uchar>(y, x) = UCHAR_MAX;
	// 		else {
	// 			if ((Pcg32::fast() % nLearningRate) == 0)
	// 				m_voBGImg[Pcg32::fast() % m_nBGSamples].at<cv::Vec3b>(y, x) = oInputImgRGB.at<cv::Vec3b>(y, x);
	// 			if ((Pcg32::fast() % nLearningRate) == 0) {
	// 				int x_rand, y_rand;
	// 				getNeighborPosition_3x3(x_rand, y_rand, x, y, m_oImgSize);
	// 				const size_t s_rand = Pcg32::fast() % m_nBGSamples;
	// 				m_voBGImg[s_rand].at<cv::Vec3b>(y_rand, x_rand) = oInputImgRGB.at<cv::Vec3b>(y, x);
	// 			}
	// 		}
	// 	}
	// }
}

void BackgroundSubtractorViBe_3ch::splitImages(const cv::Mat& inputImg, std::vector<cv::Mat>& outputImages, int numSlices) {
	outputImages.resize(numSlices);
	int y = 0;
	int h = m_oImgSize.height / numSlices;
	for (int i = 0; i < numSlices; ++i) {
		if (i == (numSlices - 1)) {
			h = m_oImgSize.height - y;
		}
		outputImages[i] = inputImg(cv::Rect(0, y, m_oImgSize.width, h));
		y += h;
		//std::cout << "h[" << i << "] = " << h << std::endl;
		//std::cout << outputImages[i].size() << std::endl;
	}
}

void BackgroundSubtractorViBe_3ch::joinImages(std::vector<cv::Mat>& inputImages, cv::Mat& outputImg) {
	int y = 0;
	for (int i = 0; i < inputImages.size(); ++i) {
		int h = inputImages[i].size().height;
		inputImages[i].copyTo(outputImg(cv::Rect(0, y, m_oImgSize.width, h)));
		y += h;
	}
}

void BackgroundSubtractorViBe_3ch::initializeParallel(const cv::Mat& initImgRGB, int numProcesses) {
	m_numProcessesParallel = numProcesses;
	m_oImgSize = initImgRGB.size();

	m_processSeq.resize(numProcesses);

	std::vector<cv::Mat> imgsSplit;
	splitImages(initImgRGB, imgsSplit, numProcesses);
	m_voBGImgParallel.resize(numProcesses);

	for (int np = 0; np < numProcesses; ++np) {
		m_processSeq[np] = np;
		//std::cout << "Process[" << np << "]" << std::endl;
		const cv::Mat& oInitImgRGB = imgsSplit[np];
		std::vector<cv::Mat> _voBGImg = std::vector<cv::Mat>(m_nBGSamples);

		const cv::Size _oImgSize = oInitImgRGB.size();
		int y_sample, x_sample;
		for (size_t s = 0; s < m_nBGSamples; s++) {
			//std::cout << "Sample[" << s << "]" << std::endl;
			_voBGImg[s].create(_oImgSize, CV_8UC3);
			_voBGImg[s] = cv::Scalar(0, 0, 0);
			for (int y_orig = 0; y_orig < _oImgSize.height; y_orig++) {
				for (int x_orig = 0; x_orig < _oImgSize.width; x_orig++) {
					lv::getSamplePosition_7x7_std2(Pcg32::fast(), x_sample, y_sample, x_orig, y_orig, 0, _oImgSize);
					_voBGImg[s].at<cv::Vec3b>(y_orig, x_orig) = oInitImgRGB.at<cv::Vec3b>(y_sample, x_sample);
				}
			}
		}
		m_voBGImgParallel[np] = _voBGImg;
	}
	m_bInitialized = true;
	//std::cout << "End initializeParallel" << std::endl;
}

void BackgroundSubtractorViBe_3ch::applyParallel(const cv::InputArray image, cv::OutputArray fgmask, double learningRate) {
	cv::Mat inputImgRGB = image.getMat();
	cv::Mat oFGMask = fgmask.getMat();
	
	std::vector<cv::Mat> imgsSplit;
	splitImages(inputImgRGB, imgsSplit, m_numProcessesParallel);

	std::vector<cv::Mat> outSplit(m_numProcessesParallel);

	std::for_each(
		std::execution::par,
		m_processSeq.begin(),
		m_processSeq.end(),
		[&](int np)
		{
			outSplit[np].create(imgsSplit[np].size(), CV_8UC1);
			applyCmp(imgsSplit[np], m_voBGImgParallel[np], outSplit[np], learningRate);
		});

	joinImages(outSplit, oFGMask);
}

void BackgroundSubtractorViBe_3ch::applyCmp(const cv::Mat& image, std::vector<cv::Mat>& bgImg, cv::Mat& fgmask, double learningRate) {
	fgmask = cv::Scalar_<uchar>(0);

	const size_t nLearningRate = (size_t)ceil(learningRate);

	cv::Size _oImgSize = image.size();

	for (int y = 0; y < _oImgSize.height; ++y) {
		for (int x = 0; x < _oImgSize.width; ++x) {
			// std::cout << "1" << std::endl;
			size_t nGoodSamplesCount = 0, nSampleIdx = 0;
			while (nGoodSamplesCount < m_nRequiredBGSamples && nSampleIdx < m_nBGSamples) {
				const cv::Vec3b& in = image.at<cv::Vec3b>(y, x);
				const cv::Vec3b& bg = bgImg[nSampleIdx].at<cv::Vec3b>(y, x);
				//if (lv::L2dist(in, bg) < m_nColorDistThreshold * 3)
				if (L2dist3Squared(in, bg) < m_nColorDistThresholdSquared)
					nGoodSamplesCount++;
				nSampleIdx++;
			}
			// std::cout << "2" << std::endl;
			if (nGoodSamplesCount < m_nRequiredBGSamples) {
				// std::cout << "3" << std::endl;
				fgmask.at<uchar>(y, x) = UCHAR_MAX;
			} else {
				// std::cout << "4" << std::endl;
				if ((Pcg32::fast() % nLearningRate) == 0)
					bgImg[Pcg32::fast() % m_nBGSamples].at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(y, x);
				if ((Pcg32::fast() % nLearningRate) == 0) {
					int x_rand, y_rand;
					getNeighborPosition_3x3(x_rand, y_rand, x, y, _oImgSize);
					const size_t s_rand = Pcg32::fast() % m_nBGSamples;
					bgImg[s_rand].at<cv::Vec3b>(y_rand, x_rand) = image.at<cv::Vec3b>(y, x);
				}
			}
		}
	}
}

// void BackgroundSubtractorViBe_3ch::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
// 	cv::Mat oInputImgRGB = _image.getMat();

// 	//_fgmask.create(m_oImgSize, CV_8UC1); // Creating outside before
// 	cv::Mat oFGMask = _fgmask.getMat();
// 	oFGMask = cv::Scalar_<uchar>(0);
	
// 	const size_t nLearningRate = (size_t)ceil(learningRate);

// 	for (int y = 0; y < m_oImgSize.height; ++y) {
// 		for (int x = 0; x < m_oImgSize.width; ++x) {
// #if BGSVIBE_USE_SC_THRS_VALIDATION
// 			const size_t nCurrSCColorDistThreshold = (size_t)(m_nColorDistThreshold * BGSVIBE_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR) / 3;
// #endif //BGSVIBE_USE_SC_THRS_VALIDATION
// 			size_t nGoodSamplesCount = 0, nSampleIdx = 0;
// 			while (nGoodSamplesCount < m_nRequiredBGSamples && nSampleIdx < m_nBGSamples) {
// 				const cv::Vec3b& in = oInputImgRGB.at<cv::Vec3b>(y, x);
// 				const cv::Vec3b& bg = m_voBGImg[nSampleIdx].at<cv::Vec3b>(y, x);
// #if BGSVIBE_USE_SC_THRS_VALIDATION
// 				for (size_t c = 0; c < 3; c++)
// 					if (lv::L1dist(in[c], bg[c]) > nCurrSCColorDistThreshold)
// 						goto skip;
// #endif //BGSVIBE_USE_SC_THRS_VALIDATION
// #if BGSVIBE_USE_L1_DISTANCE_CHECK
// 				if (lv::L1dist(in, bg) < m_nColorDistThreshold * 3)
// #else //(!BGSVIBE_USE_L1_DISTANCE_CHECK)
// 				//if (lv::L2dist(in, bg) < m_nColorDistThreshold * 3)
// 				//if (L2dist3(in, bg) < m_nColorDistThreshold * 3)
// 				if (L2dist3Squared(in, bg) < m_nColorDistThresholdSquared * 3)
// #endif //(!BGSVIBE_USE_L1_DISTANCE_CHECK)
// 					nGoodSamplesCount++;
// #if BGSVIBE_USE_SC_THRS_VALIDATION
// 				skip :
// #endif //BGSVIBE_USE_SC_THRS_VALIDATION
// 				nSampleIdx++;
// 			}
// 			if (nGoodSamplesCount < m_nRequiredBGSamples)
// 				oFGMask.at<uchar>(y, x) = UCHAR_MAX;
// 			else {
// 				if ((pcg32_fast() % nLearningRate) == 0)
// 					m_voBGImg[pcg32_fast() % m_nBGSamples].at<cv::Vec3b>(y, x) = oInputImgRGB.at<cv::Vec3b>(y, x);
// 				if ((pcg32_fast() % nLearningRate) == 0) {
// 					int x_rand, y_rand;
// 					lv::getNeighborPosition_3x3(pcg32_fast(), x_rand, y_rand, x, y, 0, m_oImgSize);
// 					const size_t s_rand = pcg32_fast() % m_nBGSamples;
// 					m_voBGImg[s_rand].at<cv::Vec3b>(y_rand, x_rand) = oInputImgRGB.at<cv::Vec3b>(y, x);
// 				}
// 			}
// 		}
// 	}
// }
