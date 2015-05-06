#ifndef GRCODER_H_
#define GRCODER_H_


#include <fstream>
#include <vector>
using namespace std;

#include "config.h"
#include "../PGMImageError.h"
#include "Bitset.h"

class GRCoder{

public:
	GRCoder(PGMImageError& input, ofstream& output);
	void encode();

private:
	PGMImageError& input;
	ofstream& output;
	Bitset data;
	unsigned log2M;

	void codeQuotient(unsigned quotient);
	void codeRemainder(unsigned remainder);
	void writeData();
};


#endif /* GRCODER_H_ */
