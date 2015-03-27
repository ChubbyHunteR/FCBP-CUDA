#ifndef PGMCBPCCUDA_H_
#define PGMCBPCCUDA_H_

#include <iostream>
#include <vector>
#include "PGMImage.h"
#include "predictors/Predictor.h"
#include "config.h"

#define CUDA_CHECK_RETURN(value) {																						\
	cudaError_t _m_cudaStat = value;																					\
	if (_m_cudaStat != cudaSuccess) {																					\
		cerr<< "Error " << cudaGetErrorString(_m_cudaStat) << " at line " << __LINE__ <<" in file " << __FILE__ << endl;\
		exit(1);																										\
	}																													\
}

struct PGMCBPCCUDA{
	PGMCBPCCUDA(PGMImage& input, PGMImage& output);
	~PGMCBPCCUDA();

	void init();
	void average();

private:
	PGMImage& input;
	PGMImage& output;
	Predictor* predictor[MAX_PREDICTORS];
	unsigned w, h, size, numOfPredictors;
	unsigned lookupOffsetx[N];
	unsigned lookupOffsety[N];
	byte* iData;
	byte* oData;

	void* dPredicted[MAX_PREDICTORS];
	void* doData;
	void* diData;
	void* dLookupOffsetx;
	void* dLookupOffsety;

	byte averagePixel(unsigned x, unsigned y);
};

#endif /* PGMCBPCCUDA_H_ */
