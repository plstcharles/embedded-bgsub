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

    void VibeBGS::initialize(const Img& _initImg) {
        initialize(_initImg, m_bgImgSamples);
    }

    void VibeBGS::initialize(const cv::Mat& _initImg) {
        Img frameImg(_initImg.data, ImgSize(_initImg.size().width, _initImg.size().height, 3));
        initialize(frameImg, m_bgImgSamples);
    }

    void VibeBGS::initializeParallel(const Img& _initImg, int _numProcesses) {
        std::vector<std::shared_ptr<Img>> imgSplit(_numProcesses);
        splitImg(_initImg, imgSplit, _numProcesses);

        m_bgImgSamplesParallel.resize(_numProcesses);
        for (int i{0}; i < _numProcesses; ++i) {
            initialize(*imgSplit[i], m_bgImgSamplesParallel[i]);
        }
    }

    void VibeBGS::initialize(const Img& _initImg, std::vector<std::shared_ptr<Img>>& _bgImgSamples) {
        int ySample, xSample;
        _bgImgSamples.resize(m_params.NBGSamples);
        for (size_t s = 0; s < m_params.NBGSamples; ++s) {
            //std::cout << "1.1" << std::endl;
            _bgImgSamples[s] = Img::create(_initImg.size, true);
            //std::cout << "1.2" << std::endl;
            for (int yOrig = 0; yOrig < _initImg.size.height; yOrig++) {
                for (int xOrig = 0; xOrig < _initImg.size.width; xOrig++) {
                    //std::cout << "1.3" << std::endl;
                    sky360::getSamplePosition_7x7_std2(Pcg32::fast(), xSample, ySample, xOrig, yOrig, _initImg.size);
                    const size_t pixelPos = (yOrig * _initImg.size.width + xOrig) * _initImg.size.numBytesPerPixel;
                    const size_t samplePos = (ySample * _initImg.size.width + xSample) * _initImg.size.numBytesPerPixel;
                    //std::cout << "1.4: " << oInitImg.size.numBytesPerPixel << std::endl;
                    _bgImgSamples[s]->data[pixelPos] = _initImg.data[samplePos];
                    _bgImgSamples[s]->data[pixelPos + 1] = _initImg.data[samplePos + 1];
                    _bgImgSamples[s]->data[pixelPos + 2] = _initImg.data[samplePos + 2];
                    //std::cout << "1.5" << std::endl;
                }
            }
        }
    }

    void VibeBGS::apply(const Img& _image, Img& _fgmask) {
        applyCmp(_image, m_bgImgSamples, _fgmask, m_params);
    }

    void VibeBGS::apply(const cv::Mat& _image, cv::Mat& _fgmask) {
        Img applyImg(_image.data, sky360::ImgSize(_image.size().width, _image.size().height, 3));
        Img maskImg(_fgmask.data, ImgSize(_fgmask.size().width, _fgmask.size().height, 1));
        applyCmp(applyImg, m_bgImgSamples, maskImg, m_params);
    }

    void VibeBGS::applyParallel(const Img& _image, Img& _fgmask) {
        // std::for_each(
        //     std::execution::par,
        //     m_processSeq.begin(),
        //     m_processSeq.end(),
        //     [&](int np)
        //     {
        //         const cv::Mat iImg{image(m_rectImgs[np])};
        //         applyCmp(iImg, m_voBGImgParallel[np], m_outSplit[np], m_params);
        //         m_outSplit[np].copyTo(fgmask(m_rectImgs[np]));
        //     });
    }

    void VibeBGS::applyCmp(const Img& image, std::vector<std::shared_ptr<Img>>& bgImg, Img& fgmask, const Params& _params) {
        //std::cout << "1.1" << std::endl;
        fgmask.clear();

        for (int pixOffset{0}, colorPixOffset{0}; 
            pixOffset < image.size.numPixels; 
            ++pixOffset, colorPixOffset += 3) {
            size_t nGoodSamplesCount{0}, 
                nSampleIdx{0};

            //std::cout << "1.2" << std::endl;
            const uchar* const pixData{&image.data[colorPixOffset]};

            while ((nGoodSamplesCount < _params.NRequiredBGSamples) 
                    && (nSampleIdx < _params.NBGSamples)) {
                //std::cout << "1.3" << std::endl;
                const uchar* const bg{&bgImg[nSampleIdx]->data[colorPixOffset]};
                if (L2dist3Squared(pixData, bg) < _params.NColorDistThresholdSquared) {
                    ++nGoodSamplesCount;
                }
                //std::cout << "1.4" << std::endl;
                ++nSampleIdx;
            }
            if (nGoodSamplesCount < _params.NRequiredBGSamples) {
                //std::cout << "1.5" << std::endl;
                fgmask.data[pixOffset] = UCHAR_MAX;
            } else {
                // if ((Pcg32::fast() % m_learningRate) == 0) {
                if ((Pcg32::fast() & _params.ANDlearningRate) == 0) {
                    //std::cout << "1.6" << std::endl;
                    uchar* const bgImgPixData{&bgImg[Pcg32::fast() & _params.ANDlearningRate]->data[colorPixOffset]};
                    bgImgPixData[0] = pixData[0];
                    bgImgPixData[1] = pixData[1];
                    bgImgPixData[2] = pixData[2];
                }
                if ((Pcg32::fast() & _params.ANDlearningRate) == 0) {
                    //std::cout << "1.7" << std::endl;
                    int neighData{getNeighborPosition_3x3(pixOffset, image.size)};
                    uchar* const xyRandData{&bgImg[Pcg32::fast() & _params.ANDlearningRate]->data[neighData * 3]};
                    xyRandData[0] = pixData[0];
                    xyRandData[1] = pixData[1];
                    xyRandData[2] = pixData[2];
                }
            }
        }
    }
}