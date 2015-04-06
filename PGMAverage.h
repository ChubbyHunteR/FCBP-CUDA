#ifndef PGMAVERAGE_H_
#define PGMAVERAGE_H_

#include <iostream>
#include "PGMImage.h"
#include "config.h"

struct PGMAverage{
	PGMAverage(PGMImage& input, PGMImage& output);

	void average();

private:
	PGMImage& input;
	PGMImage& output;
	unsigned w, h, size;
	unsigned lookupOffsetx[R_A];
	unsigned lookupOffsety[R_A];

	byte averagePixel(unsigned x, unsigned y);
};

#endif /* PGMAVERAGE_H_ */
