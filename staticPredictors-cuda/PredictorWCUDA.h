#ifndef PREDICTORWCUDA_H_
#define PREDICTORWCUDA_H_

#include "PredictorCUDA.h"

struct PredictorW : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORWCUDA_H_ */
