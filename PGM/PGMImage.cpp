#include <cstring>
#include <iostream>		
#include <iomanip>
#include <sstream>
#include <cstdlib>
using namespace std;

#include "PGMImage.h"

PGMImage::PGMImage(const char* imgName) :
		mode(ios::in), imgFile( new fstream(imgName, ios::in | ios::binary) ), imgName(imgName),
		height(0), pixelMax(0), width(0)
{
	if(imgFile->fail()){
		cerr<<"Opening of input file '"<<imgName<<"' failed."<<endl;
		exit(EXIT_FAILURE);
	}
	string tmp;
	*imgFile >> tmp;
	string magicNum(tmp);
	if (magicNum != "P5") {
        std::cerr << "Wrong file format '" << imgName << "'" << endl;
		exit(EXIT_FAILURE);
	}

	string line;
	*imgFile >> line;
	while(!isdigit(line[0])){
		*imgFile >> line;
	}

	istringstream stream(line);
	stream>> width;
	*imgFile >> height >> pixelMax;
	imgFile->ignore();

	imgBuffer = new byte[width * height];
	imgFile->read((char*)imgBuffer, width * height);
}

PGMImage::PGMImage(const char* imgName, unsigned width, unsigned height, unsigned pixelMax) :
		mode(ios::out), imgFile( new fstream(imgName, ios::binary | ios::out) ), imgName(imgName),
		width(width), height(height), pixelMax(pixelMax)
{        
	if(imgFile->fail()){
		cerr<<"Opening of output file failed."<<endl;
		exit(EXIT_FAILURE);
	}

    imgBuffer = new byte[width * height];
            
    *imgFile << "P5\n" << width << " " << height << "\n" << pixelMax << endl;
}

PGMImage::PGMImage(PGMImage&& other) :
		mode(other.mode), imgFile(other.imgFile), imgName(other.imgName),
		width(other.width), height(other.height), pixelMax(other.pixelMax),
		imgBuffer(other.imgBuffer)
{
	other.imgFile = nullptr;
	other.imgBuffer = nullptr;
}

PGMImage::~PGMImage() {
	if(imgBuffer == nullptr){
		return;
	}

    if (mode != ios::in) {
        imgFile->write( ((char*)(imgBuffer)), width * height);
    }

    delete imgFile;
    delete[] imgBuffer;
}

byte PGMImage::getPixel(unsigned x, unsigned y) {
    if(x < width && y < height) {
        return imgBuffer[x + y * width];
    }
    return 0;
}

byte PGMImage::getPixel(unsigned p) {
    if(p < width * height) {
    	return imgBuffer[p];
    }
    return 0;
}

byte* PGMImage::getBuffer() {
    return imgBuffer;
}

void PGMImage::writePixel(unsigned x, unsigned y, byte pixel) {
	if(mode != ios::out) {
    	return;
    }
    if(x < width && y < height){
    	imgBuffer[x + y * width] = pixel;
    }
}

void PGMImage::setBuffer(byte *buffer){
	if(buffer != imgBuffer){
		imgBuffer = buffer;
	}
}

void PGMImage::imageToText() {
	string txtImageName(imgName);
	txtImageName += ".txt";

	ofstream txtImageFile(txtImageName.data());
	
	txtImageFile << "Image name: " << imgName << endl;
	txtImageFile << "Width: " << width << endl;
	txtImageFile << "Height: " << height << endl;
	txtImageFile << "PIXELS" << endl;
	txtImageFile << "----------------------------------------" << endl;

	for(unsigned y = 0; y < height; ++y) {
		for(unsigned x = 0; x < width; ++x) {
			txtImageFile << std::setw(5) << imgBuffer[x + y * width];
		}
		txtImageFile << endl;
	}
}
