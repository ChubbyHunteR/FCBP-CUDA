#ifndef PGMIMAGEERROR_H
#define PGMIMAGEERROR_H

#include <fstream>
#include <string>
using namespace std;

class PGMImageError  {
public:
	// Read constructor
	PGMImageError(const char* imgName);
	// Write constructor
	PGMImageError(const char* imgName, unsigned width, unsigned height, unsigned pixelMax);
	// Copy constructor
	PGMImageError(PGMImageError&& other);

	virtual ~PGMImageError();

	unsigned getWidth() { return width; }
	unsigned getHeight() { return height; }
	unsigned getSize() { return width * height;}
	unsigned getPixelMax() { return pixelMax; }
    string getName() { return imgName; }

    short getPixel(unsigned x, unsigned y);
    short getPixel(unsigned p);
    short* getBuffer();
	void writePixel(unsigned x, unsigned y, short pixel);
	void writePixel(unsigned p, short pixel);
	void setBuffer(short *buffer);

protected:
	ios_base::openmode mode;
	fstream* imgFile;
    string imgName;

	unsigned width;
	unsigned height;
	unsigned pixelMax;

	short* imgBuffer;
};

#endif //PGMIMAGEERROR_H
