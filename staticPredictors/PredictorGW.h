#ifndef PREDICTORGW_H_
#define PREDICTORGW_H_

#include "Predictor.h"

struct PredictorGW : public Predictor{
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORGW_H_ */
