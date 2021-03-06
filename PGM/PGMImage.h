#ifndef PGMIMAGE_H
#define PGMIMAGE_H

#include <fstream>
#include <string>
using namespace std;

typedef unsigned char byte;

class PGMImage  {
public:
    // Read constructor
	PGMImage(const char* imgName);
	//Write constructor
	PGMImage(const char* imgName, unsigned width, unsigned height, unsigned pixelMax);
	// Copy constructors
	PGMImage(PGMImage&& other);

	virtual ~PGMImage();

	unsigned getWidth() { return width; }
	unsigned getHeight() { return height; }
	unsigned getSize() { return width * height;}
	unsigned getPixelMax() { return pixelMax; }
    string getName() { return imgName; }

    byte getPixel(unsigned x, unsigned y);
    byte getPixel(unsigned p);
    byte* getBuffer();
	void writePixel(unsigned x, unsigned y, byte pixel);
	void writePixel(unsigned p, byte pixel);
	void setBuffer(byte* buffer);

	void imageToText();
	void invalidate();

protected:
	ios_base::openmode mode;
    fstream* imgFile;
    string imgName;

	unsigned width;
	unsigned height;
	unsigned pixelMax;

	byte* imgBuffer;
};

#endif //PGMIMAGE_H
