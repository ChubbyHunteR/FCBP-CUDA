#ifndef PREDICTORN_H_
#define PREDICTORN_H_

#include "Predictor.h"

struct PredictorN : public Predictor{
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORN_H_ */
