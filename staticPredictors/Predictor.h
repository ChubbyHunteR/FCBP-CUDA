#ifndef PREDICTOR_H_
#define PREDICTOR_H_
typedef unsigned char byte;

struct Predictor{
	virtual byte predict(void *iData, unsigned x, unsigned y, unsigned w, unsigned h) = 0;
	virtual void predictAll(void *iData, void *predicted, unsigned w, unsigned h);
};

#endif /* PREDICTOR_H_ */
