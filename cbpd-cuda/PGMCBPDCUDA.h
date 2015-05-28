#ifndef PGMCBPDCUDA_H_
#define PGMCBPDCUDA_H_

#include <vector>
#include <cuda_runtime.h>

#include "../PGM/PGMImage.h"
#include "../PGM/PGMImageError.h"
#include "../config.h"
#include "../util.h"

#define CUDA_CHECK_RETURN(value) {																						\
	cudaError_t _m_cudaStat = value;																					\
	if (_m_cudaStat != cudaSuccess) {																					\
		cerr<< "Error " << cudaGetErrorString(_m_cudaStat) << " at line " << __LINE__ <<" in file " << __FILE__ << endl;\
		exit(1);																										\
	}																													\
}

struct PGMCBPDCUDA{
	PGMCBPDCUDA(vector<PGMImageError>& inputImagesError,
				vector<PGMImage>& outputImages
				);
	~PGMCBPDCUDA();

	void decode();

private:
	vector<PGMImageError>& inputImagesError;
	vector<PGMImage>& outputImages;

	vector<ImageWHSize> imagesMeta;
	vector<short*> iData;
	vector<byte*> oData;

	PixelOffset radiusOffset[R_A];
	PixelOffset vectorOffset[D];

	void* diData;
	void* doData;
	void* dpMemoData;
	void* dpData;
	void* dImagesMeta;
	void* dRadiusOffset;
	void* dVectorOffset;
};

#endif /* PGMCBPDCUDA_H_ */
