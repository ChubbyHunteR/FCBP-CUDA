#ifndef PREDICTORW_H_
#define PREDICTORW_H_

#include "Predictor.h"

struct PredictorW : public Predictor{
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORW_H_ */
