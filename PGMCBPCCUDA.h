#ifndef PGMCBPCCUDA_H_
#define PGMCBPCCUDA_H_

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
	PGMCBPCCUDA(PGMImage& input, PGMImage& output, PGMImage& outputError);
	~PGMCBPCCUDA();

	bool init();
	bool addPredictor(Predictor* predictor);
	void predict();

	bool getStaticPrediction(unsigned i);

private:
	bool locked;

	PGMImage& input;
	PGMImage& output;
	PGMImage& outputError;
	std::vector<Predictor*> predictors;
	unsigned w, h, size;
	unsigned radiusOffsetx[R_A];
	unsigned radiusOffsety[R_A];
	unsigned vectorOffsetx[D];
	unsigned vectorOffsety[D];
	byte* iData;
	byte* oData;
	int* predictionError;

	void** dPredicted;
	void* dRadiusOffsetx;
	void* dRadiusOffsety;
	void* dVectorOffsetx;
	void* dVectorOffsety;
	void* diData;
	void* doData;
	void* dPredictionError;
};

#endif /* PGMCBPCCUDA_H_ */
