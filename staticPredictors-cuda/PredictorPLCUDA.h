#ifndef PREDICTORPLCUDA_H_
#define PREDICTORPLCUDA_H_

#include "PredictorCUDA.h"

struct PredictorPL : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
};

#endif /* PREDICTORPLCUDA_H_ */
