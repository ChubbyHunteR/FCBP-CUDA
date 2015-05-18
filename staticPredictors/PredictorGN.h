#ifndef PREDICTORGN_H_
#define PREDICTORGN_H_

#include "Predictor.h"

struct PredictorGN : public Predictor{
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h);
};

#endif /* PREDICTORGN_H_ */
