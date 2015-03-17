#ifndef PGMAVERAGECUDA_H_
#define PGMAVERAGECUDA_H_

#include <iostream>
#include "PGMImage.h"
#include "config.h"

#define CUDA_CHECK_RETURN(value) {																						\
	cudaError_t _m_cudaStat = value;																					\
	if (_m_cudaStat != cudaSuccess) {																					\
		cerr<< "Error " << cudaGetErrorString(_m_cudaStat) << " at line " << __LINE__ <<" in file " << __FILE__ << endl;\
		exit(1);																										\
	}																													\
}

struct PGMAverageCUDA{
	PGMAverageCUDA(PGMImage& input, PGMImage& output);
	~PGMAverageCUDA();

	void average();

private:
	PGMImage& input;
	PGMImage& output;
	unsigned w, h, size;
	unsigned lookupOffsetx[N];
	unsigned lookupOffsety[N];
	byte* iData;
	byte* oData;

	void* doData;
	void* diData;
	void* dLookupOffsetx;
	void* dLookupOffsety;

	byte averagePixel(unsigned x, unsigned y);
};

#endif /* PGMAVERAGECUDA_H_ */
