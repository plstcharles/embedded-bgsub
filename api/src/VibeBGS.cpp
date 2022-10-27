#include "VibeBGS.hpp"

#include <iostream>
#include <execution>

namespace sky360 {

    VibeBGS::VibeBGS(size_t _nColorDistThreshold,
                size_t _nBGSamples,
                size_t _nRequiredBGSamples,
                size_t _learningRate)
        : m_params(_nColorDistThreshold, _nBGSamples, _nRequiredBGSamples, _learningRate)
    {}

    void VibeBGS::initialize(const cv::Mat& _initImg, int _numProcesses) {
        m_numProcessesParallel = _numProcesses;

        std::vector<std::shared_ptr<Img>> imgSplit(_numProcesses);
        Img frameImg(_initImg.data, ImgSize(_initImg.size().width, _initImg.size().height, _initImg.channels()));
        splitImg(frameImg, imgSplit, _numProcesses);

        m_processSeq.resize(_numProcesses);
        m_bgImgSamples.resize(_numProcesses);

        for (int i{0}; i < _numProcesses; ++i) {
            m_processSeq[i] = i;
            initialize(*imgSplit[i], m_bgImgSamples[i]);
        }
    }

    void VibeBGS::initialize(const Img& _initImg, std::vector<std::unique_ptr<Img>>& _bgImgSamples) {
        Pcg32 pcg32;
        int ySample, xSample;
        _bgImgSamples.resize(m_params.NBGSamples);
        for (size_t s{0}; s < m_params.NBGSamples; ++s) {
            _bgImgSamples[s] = Img::create(_initImg.size, false);
            for (int yOrig{0}; yOrig < _initImg.size.height; yOrig++) {
                for (int xOrig{0}; xOrig < _initImg.size.width; xOrig++) {
                    getSamplePosition_7x7_std2(pcg32.fast(), xSample, ySample, xOrig, yOrig, _initImg.size);
                    const size_t pixelPos = (yOrig * _initImg.size.width + xOrig) * _initImg.size.numBytesPerPixel;
                    const size_t samplePos = (ySample * _initImg.size.width + xSample) * _initImg.size.numBytesPerPixel;
                    _bgImgSamples[s]->data[pixelPos] = _initImg.data[samplePos];
                    if (_initImg.size.numBytesPerPixel > 1) {
                        _bgImgSamples[s]->data[pixelPos + 1] = _initImg.data[samplePos + 1];
                        _bgImgSamples[s]->data[pixelPos + 2] = _initImg.data[samplePos + 2];
                    }
                }
            }
        }
    }

    void VibeBGS::apply(const cv::Mat& _image, cv::Mat& _fgmask) {
        Img applyImg(_image.data, ImgSize(_image.size().width, _image.size().height, _image.channels()));
        Img maskImg(_fgmask.data, ImgSize(_fgmask.size().width, _fgmask.size().height, 1));
        if (m_numProcessesParallel == 1) {
            if (_image.channels() > 1)
                apply3(applyImg, m_bgImgSamples[0], maskImg, m_params);
            else
                apply1(applyImg, m_bgImgSamples[0], maskImg, m_params);
        } else {
            applyParallel(applyImg, maskImg);
        }
    }

    void VibeBGS::applyParallel(const Img& _image, Img& _fgmask) {
        std::for_each(
            std::execution::par,
            m_processSeq.begin(),
            m_processSeq.end(),
            [&](int np)
            {
                Img imgSplit(_image.data + (m_bgImgSamples[np][0]->size.originalPixelPos * m_bgImgSamples[np][0]->size.numBytesPerPixel), 
                            ImgSize(m_bgImgSamples[np][0]->size.width, m_bgImgSamples[np][0]->size.height, _image.size.numBytesPerPixel));
                Img maskPartial(_fgmask.data + m_bgImgSamples[np][0]->size.originalPixelPos, 
                            ImgSize(imgSplit.size.width, imgSplit.size.height, 1));
                if (_image.size.numBytesPerPixel > 1)
                    apply3(imgSplit, m_bgImgSamples[np], maskPartial, m_params);
                else
                    apply1(imgSplit, m_bgImgSamples[np], maskPartial, m_params);
            });
    }

    void VibeBGS::apply3(const Img& _image, std::vector<std::unique_ptr<Img>>& _bgImg, Img& _fgmask, const Params& _params) {
        Pcg32 pcg32;
        _fgmask.clear();

        for (int pixOffset{0}, colorPixOffset{0}; 
                pixOffset < _image.size.numPixels; 
                ++pixOffset, colorPixOffset += _image.size.numBytesPerPixel) {
            size_t nGoodSamplesCount{0}, 
                nSampleIdx{0};

            const uchar* const pixData{&_image.data[colorPixOffset]};

            while (nSampleIdx < _params.NBGSamples) {
                const uchar* const bg{&_bgImg[nSampleIdx]->data[colorPixOffset]};
                if (L2dist3Squared(pixData, bg) < _params.NColorDistThresholdSquared) {
                    ++nGoodSamplesCount;
                    if (nGoodSamplesCount >= _params.NRequiredBGSamples) {
                        // if ((Pcg32::fast() % m_learningRate) == 0) {
                        if ((pcg32.fast() & _params.ANDlearningRate) == 0) {
                            uchar* const bgImgPixData{&_bgImg[pcg32.fast() & _params.ANDlearningRate]->data[colorPixOffset]};
                            bgImgPixData[0] = pixData[0];
                            bgImgPixData[1] = pixData[1];
                            bgImgPixData[2] = pixData[2];
                        }
                        if ((pcg32.fast() & _params.ANDlearningRate) == 0) {
                            const int neighData{getNeighborPosition_3x3(pixOffset, _image.size, pcg32) * 3};
                            uchar* const xyRandData{&_bgImg[pcg32.fast() & _params.ANDlearningRate]->data[neighData]};
                            xyRandData[0] = pixData[0];
                            xyRandData[1] = pixData[1];
                            xyRandData[2] = pixData[2];
                        }
                        break;
                    }
                }
                ++nSampleIdx;
            }
            if (nGoodSamplesCount < _params.NRequiredBGSamples) {
                _fgmask.data[pixOffset] = UCHAR_MAX;
            } 
        }
    }

    void VibeBGS::apply1(const Img& _image, std::vector<std::unique_ptr<Img>>& _bgImg, Img& _fgmask, const Params& _params) {
        Pcg32 pcg32;
        _fgmask.clear();

        for (int pixOffset{0}, colorPixOffset{0}; 
                pixOffset < _image.size.numPixels; 
                ++pixOffset, colorPixOffset += _image.size.numBytesPerPixel) {
            size_t nGoodSamplesCount{0}, 
                nSampleIdx{0};

            const uchar* const pixData{&_image.data[colorPixOffset]};

            while (nSampleIdx < _params.NBGSamples) {
                const uchar* const bg{&_bgImg[nSampleIdx]->data[colorPixOffset]};
                if (L1dist(pixData, bg) < _params.NColorDistThreshold) {
                    ++nGoodSamplesCount;
                    if (nGoodSamplesCount >= _params.NRequiredBGSamples) {
                        if ((pcg32.fast() & _params.ANDlearningRate) == 0) {
                            uchar* const bgImgPixData{&_bgImg[pcg32.fast() & _params.ANDlearningRate]->data[colorPixOffset]};
                            bgImgPixData[0] = pixData[0];
                        }
                        if ((pcg32.fast() & _params.ANDlearningRate) == 0) {
                            const int neighData{getNeighborPosition_3x3(pixOffset, _image.size, pcg32)};
                            uchar* const xyRandData{&_bgImg[pcg32.fast() & _params.ANDlearningRate]->data[neighData]};
                            xyRandData[0] = pixData[0];
                        }
                        break;
                    }
                }
                ++nSampleIdx;
            }
            if (nGoodSamplesCount < _params.NRequiredBGSamples) {
                _fgmask.data[pixOffset] = UCHAR_MAX;
            }
            // else {
            //     // if ((Pcg32::fast() % m_learningRate) == 0) {
            //     if ((Pcg32::fast() & _params.ANDlearningRate) == 0) {
            //         uchar* const bgImgPixData{&bgImg[Pcg32::fast() & _params.ANDlearningRate]->data[colorPixOffset]};
            //         bgImgPixData[0] = pixData[0];
            //         bgImgPixData[1] = pixData[1];
            //         bgImgPixData[2] = pixData[2];
            //     }
            //     if ((Pcg32::fast() & _params.ANDlearningRate) == 0) {
            //         int neighData{getNeighborPosition_3x3(pixOffset, image.size)};
            //         uchar* const xyRandData{&bgImg[Pcg32::fast() & _params.ANDlearningRate]->data[neighData * 3]};
            //         xyRandData[0] = pixData[0];
            //         xyRandData[1] = pixData[1];
            //         xyRandData[2] = pixData[2];
            //     }
            // }
        }
    }

    void VibeBGS::getBackgroundImage(cv::Mat& backgroundImage) const {
        // cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize, CV_32FC(m_bgImgSamples[0].size.numBytesPerPixel));
        // for(size_t n = 0; n < m_nBGSamples; ++n) {
        //     for(int y = 0; y < m_oImgSize.height; ++y) {
        //         for(int x = 0; x < m_oImgSize.width; ++x) {
        //             const size_t idx_uchar = m_voBGImg[n].step.p[0] * y + m_voBGImg[n].step.p[1] * x;
        //             const size_t idx_flt32 = idx_uchar * 4;
        //             float* oAvgBgImgPtr = (float*)(oAvgBGImg.data + idx_flt32);
        //             const uchar* const oBGImgPtr = m_voBGImg[n].data + idx_uchar;
        //             for(size_t c = 0; c < (size_t)m_voBGImg[n].channels(); ++c)
        //                 oAvgBgImgPtr[c] += ((float)oBGImgPtr[c]) / m_nBGSamples;
        //         }
        //     }
        // }
        // oAvgBGImg.convertTo(backgroundImage, CV_8U);
    }
}