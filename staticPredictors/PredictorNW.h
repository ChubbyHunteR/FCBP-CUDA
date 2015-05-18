#ifndef PREDICTORNW_H_
#define PREDICTORNW_H_

#include "Predictor.h"

struct PredictorNW : public Predictor{
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORNW_H_ */
