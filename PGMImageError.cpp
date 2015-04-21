#include <cstring>
#include <iostream>		
#include <iomanip>
#include <sstream>
#include <cstdlib>
using namespace std;

#include "PGMImageError.h"


PGMImageError::PGMImageError(const char* imgName) :
		mode(ios::in), imgFile( new fstream(imgName, ios::in | ios::binary) ), imgName(imgName),
		width(0), height(0), pixelMax(0)
{
	if(imgFile->fail()){
		cerr<<"Opening of input file "<<imgName<<" failed."<<endl;
		exit(EXIT_FAILURE);
	}

	*imgFile >> width >> height >> pixelMax;

	imgBuffer = new short[width * height];
	imgFile->read((char*)imgBuffer, width * height * sizeof(short));
}

PGMImageError::PGMImageError(const char* imgName, unsigned width, unsigned height, unsigned pixelMax) :
		mode(ios::out), imgFile( new fstream(imgName, ios::binary | ios::out) ),
		width(width), height(height), pixelMax(pixelMax), imgName(imgName)
{
	if(imgFile->fail()){
		cerr<<"Opening of output file failed."<<endl;
		exit(EXIT_FAILURE);
	}

    imgBuffer = new short[width * height];
            
    *imgFile << width << height << pixelMax;
    
}

PGMImageError::PGMImageError(PGMImageError&& other) :
		mode(other.mode), imgFile(other.imgFile), width(other.width), imgName(other.imgName),
		height(other.height), pixelMax(other.pixelMax), imgBuffer(other.imgBuffer)
{
	other.imgFile = nullptr;
	other.imgBuffer = nullptr;
}

PGMImageError::~PGMImageError() {
	if(imgBuffer == nullptr){
		return;
	}

    if (mode != ios::in) {
        imgFile->write((char*)imgBuffer, width * height * sizeof(short));
    }

    delete imgFile;
    delete[] imgBuffer;
}

short PGMImageError::getPixel(unsigned x, unsigned y) {
    if(x < width && y < height) {
        return imgBuffer[x + y * width];
    }
    return 0;
}

short PGMImageError::getPixel(unsigned p) {
    if(p < width * height) {
    	return imgBuffer[p];
    }
    return 0;
}

short* PGMImageError::getBuffer() {
    return imgBuffer;
}

void PGMImageError::writePixel(unsigned x, unsigned y, short pixel) {
    if(mode != ios::out) {
    	return;
    }

    if (x < width && y < height) {
    	imgBuffer[x + y * width] = pixel;
    }
}

void PGMImageError::writePixel(unsigned p, short pixel) {
    if(mode != ios::out) {
    	return;
    }

    if (p < width * height) {
    	imgBuffer[p] = pixel;
    }
}
