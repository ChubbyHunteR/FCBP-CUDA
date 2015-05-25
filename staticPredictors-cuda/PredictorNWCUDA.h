#ifndef PREDICTORNWCUDA_H_
#define PREDICTORNWCUDA_H_

#include "PredictorCUDA.h"

struct PredictorNW : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORNWCUDA_H_ */
