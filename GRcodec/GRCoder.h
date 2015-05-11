#ifndef GRCODER_H_
#define GRCODER_H_

#include <fstream>
#include <vector>

#include "GRconfig.h"
#include "../PGM/PGMImageError.h"
#include "Bitset.h"

using namespace std;

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
