#ifndef PREDICTORS_H_
#define PREDICTORS_H_

typedef unsigned char byte;

__device__ byte predictGN(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
__device__ byte predictGW(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
__device__ byte predictN(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
__device__ byte predictNE(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
__device__ byte predictNW(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
__device__ byte predictPL(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
__device__ byte predictW(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);

extern __device__ unsigned numOfPredictors;

extern __device__ byte (* predictors[])(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);

#endif /* PREDICTORS_H_ */
