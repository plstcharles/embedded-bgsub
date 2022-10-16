#pragma once

#include <iostream>
#include <typeinfo>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define NEXT_BIGGER_INTEGER(curr, next) \
    template<> \
    struct get_bigger_integer<curr> {\
        typedef next type; \
    }

namespace lv {

	/// type traits helper which provides the smallest integer type that is bigger than Tint
	template<typename Tint>
	struct get_bigger_integer {
	};
	NEXT_BIGGER_INTEGER(uint8_t, uint16_t);

	/// returns pixel coordinates clamped to the given image & border size
	inline void clampImageCoords(int& nSampleCoord_X, int& nSampleCoord_Y, const cv::Size& oImageSize) {
		if (nSampleCoord_X < 0)
			nSampleCoord_X = 0;
		else if (nSampleCoord_X >= oImageSize.width)
			nSampleCoord_X = oImageSize.width - 1;
		if (nSampleCoord_Y < 0)
			nSampleCoord_Y = 0;
		else if (nSampleCoord_Y >= oImageSize.height)
			nSampleCoord_Y = oImageSize.height - 1;
	}

	/// returns the sampling location for the specified random index & original pixel location, given a predefined kernel; also guards against out-of-bounds values via image/border size check
	template<int nKernelHeight, int nKernelWidth>
	inline void getSamplePosition(const std::array<std::array<int, nKernelWidth>, nKernelHeight>& anSamplesInitPattern,
		const int nSamplesInitPatternTot, const int nRandIdx, int& nSampleCoord_X, int& nSampleCoord_Y,
		const int nOrigCoord_X, const int nOrigCoord_Y, const cv::Size& oImageSize) {
		int r = 1 + (nRandIdx % nSamplesInitPatternTot);
		for (nSampleCoord_Y = 0; nSampleCoord_Y < nKernelHeight; ++nSampleCoord_Y) {
			for (nSampleCoord_X = 0; nSampleCoord_X < nKernelWidth; ++nSampleCoord_X) {
				r -= anSamplesInitPattern[nSampleCoord_Y][nSampleCoord_X];
				if (r <= 0)
					goto stop;
			}
		}
	stop:
		nSampleCoord_X += nOrigCoord_X - nKernelWidth / 2;
		nSampleCoord_Y += nOrigCoord_Y - nKernelHeight / 2;
		clampImageCoords(nSampleCoord_X, nSampleCoord_Y, oImageSize);
	}

	/// returns the sampling location for the specified random index & original pixel location; also guards against out-of-bounds values via image/border size check
	inline void getSamplePosition_7x7_std2(const int nRandIdx, int& nSampleCoord_X, int& nSampleCoord_Y,
		const int nOrigCoord_X, const int nOrigCoord_Y,
		const int nBorderSize, const cv::Size& oImageSize) {
		// based on 'floor(fspecial('gaussian',7,2)*512)'
		static const int s_nSamplesInitPatternTot = 512;
		static const std::array<std::array<int, 7>, 7> s_anSamplesInitPattern = {
				std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
				std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
				std::array<int,7>{ 6,12,21,25,21,12, 6,},
				std::array<int,7>{ 7,14,25,28,25,14, 7,},
				std::array<int,7>{ 6,12,21,25,21,12, 6,},
				std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
				std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
		};
		getSamplePosition<7, 7>(s_anSamplesInitPattern, s_nSamplesInitPatternTot, nRandIdx, nSampleCoord_X, nSampleCoord_Y, nOrigCoord_X, nOrigCoord_Y, oImageSize);
	}

	// /// returns the neighbor location for the specified random index & original pixel location, given a predefined neighborhood; also guards against out-of-bounds values via image/border size check
	// template<int nNeighborCount>
	// inline void getNeighborPosition(const std::array<std::array<int, 2>, nNeighborCount>& anNeighborPattern,
	// 	const int nRandIdx, int& nNeighborCoord_X, int& nNeighborCoord_Y,
	// 	const int nOrigCoord_X, const int nOrigCoord_Y, const cv::Size& oImageSize) {
	// 	const int r = nRandIdx % nNeighborCount;
	// 	nNeighborCoord_X = nOrigCoord_X + anNeighborPattern[r][0];
	// 	nNeighborCoord_Y = nOrigCoord_Y + anNeighborPattern[r][1];
	// 	clampImageCoords(nNeighborCoord_X, nNeighborCoord_Y, oImageSize);
	// }

	// /// returns the neighbor location for the specified random index & original pixel location; also guards against out-of-bounds values via image/border size check
	// inline void getNeighborPosition_3x3(const int nRandIdx, int& nNeighborCoord_X, int& nNeighborCoord_Y,
	// 	const int nOrigCoord_X, const int nOrigCoord_Y, const cv::Size& oImageSize) {
	// 	typedef std::array<int, 2> Nb;
	// 	static const std::array<std::array<int, 2>, 8> s_anNeighborPattern = {
	// 			Nb{-1, 1},Nb{0, 1},Nb{1, 1},
	// 			Nb{-1, 0},         Nb{1, 0},
	// 			Nb{-1,-1},Nb{0,-1},Nb{1,-1},
	// 	};
	// 	const int r = nRandIdx % 8;
	// 	nNeighborCoord_X = nOrigCoord_X + s_anNeighborPattern[r][0];
	// 	nNeighborCoord_Y = nOrigCoord_Y + s_anNeighborPattern[r][1];
	// 	clampImageCoords(nNeighborCoord_X, nNeighborCoord_Y, oImageSize);
	// }

	// /// returns the neighbor location for the specified random index & original pixel location; also guards against out-of-bounds values via image/border size check
	// inline void getNeighborPosition_3x3(const int nRandIdx, int& nNeighborCoord_X, int& nNeighborCoord_Y,
	// 	const int nOrigCoord_X, const int nOrigCoord_Y, const cv::Size& oImageSize) {
	// 	typedef std::array<int, 2> Nb;
	// 	static const std::array<std::array<int, 2>, 8> s_anNeighborPattern = {
	// 			Nb{-1, 1},Nb{0, 1},Nb{1, 1},
	// 			Nb{-1, 0},         Nb{1, 0},
	// 			Nb{-1,-1},Nb{0,-1},Nb{1,-1},
	// 	};
	// 	getNeighborPosition<8>(s_anNeighborPattern, nRandIdx, nNeighborCoord_X, nNeighborCoord_Y, nOrigCoord_X, nOrigCoord_Y, oImageSize);
	// }

	/// computes the squared L2 distance between two integer values (i.e. == squared L1dist); returns an unsigned type of twice the size of the input type
	template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
	inline auto L2sqrdist(T a, T b) {
		typedef std::make_signed_t<typename lv::get_bigger_integer<T>::type> Tintern;
		//std::cout << "T: " << typeid(T).name() << ", Tintern: " << typeid(Tintern).name() << "\n";
		const Tintern tResult = Tintern(a) - Tintern(b);
		return std::make_unsigned_t<Tintern>(tResult * tResult);
	}

	/// computes the L1 distance between two integer values; returns an unsigned type of the same size as the input type
	template<typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
	inline auto L1dist(T a, T b) {
		return (std::make_unsigned_t<T>)std::abs(((std::make_signed_t<typename lv::get_bigger_integer<T>::type>)a) - b);
	}

	/// computes the L2 distance between two integer arrays
	template<size_t nChannels, typename Tin, typename Tout = float, typename = std::enable_if_t<std::is_integral<Tin>::value>>
	inline Tout L2dist(const Tin* a, const Tin* b) {
		decltype(L2sqrdist(Tin(), Tin())) tResult = 0;
		//std::cout << "Tin: " << typeid(Tin).name() << ", TResult: " << typeid(tResult).name() << "\n";
		for (size_t c = 0; c < nChannels; ++c)
			tResult += L2sqrdist(a[c], b[c]);
		return (Tout)std::sqrt(tResult);
	}

	// /// computes the L2 distance between two opencv vectors
	// // Modified
	// template<int nChannels, typename Tin, typename Tout = decltype(L2dist<nChannels>((Tin*)0, (Tin*)0))>
	// inline Tout L2dist(const cv::Vec<Tin, nChannels>& a, const cv::Vec<Tin, nChannels>& b) {
	// 	//std::cout << "Tout: " << typeid(Tout).name() << "\n";
	// 	decltype(L2sqrdist(Tin(), Tin())) tResult = 0;
	// 	for (size_t c = 0; c < nChannels; ++c)
	// 		tResult += (ushort)(short(a[c]) - short(b[c]));
	// 		//tResult += L2sqrdist(a[c], b[c]);
	// 	return (Tout)std::sqrt(tResult);
	// }

	// /// computes the L2 distance between two opencv vectors
	// template<int nChannels, typename Tin, typename Tout = decltype(L2dist<nChannels>((Tin*)0, (Tin*)0))>
	// inline Tout L2dist(const cv::Vec<Tin, nChannels>& a, const cv::Vec<Tin, nChannels>& b) {
	// 	Tin a_array[nChannels], b_array[nChannels];
	// 	for (int c = 0; c < nChannels; ++c) {
	// 		a_array[c] = a[c];
	// 		b_array[c] = b[c];
	// 	}
	// 	return (Tout)L2dist<nChannels>(a_array, b_array);
	// }
}