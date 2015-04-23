#ifndef PREDICTORNE_H_
#define PREDICTORNE_H_

#include "Predictor.h"

struct PredictorNE : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORNE_H_ */
