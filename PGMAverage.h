#ifndef PGMAVERAGE_H_
#define PGMAVERAGE_H_

#include <iostream>
#include "PGMImage.h"

#define CUDA_CHECK_RETURN(value) {																						\
	cudaError_t _m_cudaStat = value;																					\
	if (_m_cudaStat != cudaSuccess) {																					\
		cerr<< "Error " << cudaGetErrorString(_m_cudaStat) << " at line " << __LINE__ <<" in file " << __FILE__ << endl;\
		exit(1);																										\
	}																													\
}

#define THREADS 512

/*
 * R defines the radius in number of pixels. Radius is number of pixels left, right and top from the "current" pixel taken into
 * account when calculating the average. All the taken pixels form an area of N pixels, equal to (R+1) times (2R+1) minus R.
 */
#define R 10
#define N (2 * R * (R  + 1) + 1)

struct PGMAverage{
	PGMAverage(PGMImage& input, PGMImage& output);

	void average();

private:
	PGMImage& input;
	PGMImage& output;
	unsigned w, h, size;
	unsigned lookupOffsetx[N];
	unsigned lookupOffsety[N];
	byte* iData;
	byte* oData;
	void* dData;

	byte averagePixel(unsigned x, unsigned y);
};

#endif /* PGMAVERAGE_H_ */
