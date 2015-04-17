#ifndef PREDICTORPL_H_
#define PREDICTORPL_H_

#include "Predictor.h"

struct PredictorPL : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORPL_H_ */
