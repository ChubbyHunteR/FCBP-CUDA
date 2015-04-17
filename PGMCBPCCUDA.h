#ifndef PGMCBPCCUDA_H_
#define PGMCBPCCUDA_H_

#include <vector>
#include "PGMImage.h"
#include "predictors/Predictor.h"
#include "config.h"
#include "util.h"

#define CUDA_CHECK_RETURN(value) {																						\
	cudaError_t _m_cudaStat = value;																					\
	if (_m_cudaStat != cudaSuccess) {																					\
		cerr<< "Error " << cudaGetErrorString(_m_cudaStat) << " at line " << __LINE__ <<" in file " << __FILE__ << endl;\
		exit(1);																										\
	}																													\
}

struct PGMCBPCCUDA{
	PGMCBPCCUDA(vector<PGMImage>& inputImages,
				vector<PGMImage>& outputImages,
				vector<PGMImage>& errorImages,
				vector<Predictor*>& predictors
				);
	~PGMCBPCCUDA();

	void predict();
	bool getStaticPrediction(unsigned i);

private:
	vector<PGMImage>& inputImages;
	vector<PGMImage>& outputImages;
	vector<PGMImage>& errorImages;
	vector<Predictor*>& predictors;

	vector<ImageWHSize> imagesMeta;
	vector<byte*> iData;
	vector<byte*> oData;
	vector<short*> eData;

	PixelOffset radiusOffset[R_A];
	PixelOffset vectorOffset[D];

	vector<void*> diData;
	vector<void*> dpData;
	vector<void*> doData;
	vector<void*> deData;
	vector<void**> dPredicted;
	void* dRadiusOffset;
	void* dVectorOffset;
};

#endif /* PGMCBPCCUDA_H_ */
