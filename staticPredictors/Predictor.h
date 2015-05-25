#ifndef PREDICTOR_H_
#define PREDICTOR_H_
typedef unsigned char byte;

struct Predictor{
	virtual byte predict(byte *iData, unsigned x, unsigned y, unsigned w, unsigned h) = 0;
	virtual void predictAll(byte *iData, byte *predicted, unsigned w, unsigned h);
};

#endif /* PREDICTOR_H_ */
