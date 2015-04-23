#ifndef PREDICTORW_H_
#define PREDICTORW_H_

#include "Predictor.h"

struct PredictorW : public Predictor{
	virtual void cudaPredictAll(void *diData, void *dPredicted, unsigned w, unsigned h);
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORW_H_ */
