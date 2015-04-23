#ifndef PGMCBPDCUDA_H_
#define PGMCBPDCUDA_H_

#include <vector>
#include "PGMImage.h"
#include "PGMImageError.h"
#include "coderPredictors/Predictor.h"
#include "config.h"
#include "util.h"

#define CUDA_CHECK_RETURN(value) {																						\
	cudaError_t _m_cudaStat = value;																					\
	if (_m_cudaStat != cudaSuccess) {																					\
		cerr<< "Error " << cudaGetErrorString(_m_cudaStat) << " at line " << __LINE__ <<" in file " << __FILE__ << endl;\
		exit(1);																										\
	}																													\
}

struct PGMCBPDCUDA{
	PGMCBPDCUDA(vector<PGMImageError>& inputImagesError,
				vector<PGMImage>& outputImages,
				vector<PGMImage>& predictionImages,
				vector<Predictor*>& predictors
				);
	~PGMCBPDCUDA();

	void decode();

private:
	vector<PGMImageError>& inputImagesError;
	vector<PGMImage>& outputImages;
	vector<PGMImage>& predictionImages;
	vector<Predictor*>& predictors;

	vector<cudaStream_t> streams;
	vector<ImageWHSize> imagesMeta;
	vector<short*> iData;
	vector<byte*> oData;
	vector<byte*> pData;

	PixelOffset radiusOffset[R_A];
	PixelOffset vectorOffset[D];

	vector<void*> diData;
	vector<void*> doData;
	vector<void*> dpData;
	void* dRadiusOffset;
	void* dVectorOffset;
};

#endif /* PGMCBPDCUDA_H_ */
