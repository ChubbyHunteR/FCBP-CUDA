///////////////////////////////////////////////////////////////////////////////
// File Name    : PGMImage.h
//                                                                          
// Class        : PGMImage
//                                                                          
// Purpose      : Interface to read/write pgm (portable gray map) image file.
//                When reading, the pixels are buffered into memory. When
//                used for writing, pixels are first saved (buffered) into 
//                memory, and in the destructor they are flushed into properly
//                formatted pgm file.
//
// Notes        : Doesn't fully support pgm format (such as pgm comments etc.)
//
// Author       : Josip Knezovic josip.knezovic@fer.hr
//
// Copyright 2005. Josip Knezovic                                            
// 
///////////////////////////////////////////////////////////////////////////////

#ifndef PGMIMAGE_H
#define PGMIMAGE_H

#include <fstream>
#include <string>
using namespace std;

typedef unsigned char byte;

class PGMImage  {
public:
    // constructors
    // This constructor is used when the PGMImage object is used for reading
    // from pgm image file
	PGMImage(const char* imgName);
	
	// Used when the PGMImage object is used to write into pgm image file
	PGMImage(const char* imgName, unsigned width, unsigned height, byte pixelMax);
	
    // Emplace back needs the copy constructor
	PGMImage(PGMImage&& other);

    //destructor
	virtual ~PGMImage();

    // returns the image width
	unsigned getWidth() { return widthM; }

    // returns the image height
	unsigned getHeight() { return heightM; }

	// returns the image size in pixels
	unsigned getSize() { return widthM * heightM;}

    // returns maximum pixel value
	byte getPixelMax() { return pixelMaxM; }

    // returns image name
    string getName() { return imgNameM; }

    // returns the pixel at the (xPos, yPos) coordinates
    byte getPixel(unsigned xPos, unsigned yPos);
    
    // returns the pixel at the pPos position when the image is viewed as 
    // an array of pixels in raster scan order
    byte getPixel(unsigned pPos);

    // returns the pixel buffer
    byte* getBuffer();

    // takes the pixels from the (xPos, yPos) to (xPos + size -1, yPos) in the
    // pgm image and fills the buff array
	void readIntoArray(byte* buff, unsigned xPos, unsigned yPos, unsigned size);

    // takes the pixels from the xPos to xPos + size -1 in the
    // pgm image (when viewed as a single array) and fills the buff array
	void readIntoArray(byte* buff, unsigned xPos, unsigned size);
	
	// writes the pixel into the xPos, yPos coordinates
	// only possible in out mode
	void writePixel(unsigned xPos, unsigned yPos, byte pixel);

    /////////////////////////////////////////////////////////////////////////
    // Function: imageToText()
	// Purpose: Outputs the image into formatted humman readable file
	//          for inspection and debugging
	// Note:    This function spills the contents of the image buffer array into
	//          text file. If this array is uninitalized in the case of out mode
    //          (if the output pgm image is not written completely) then the
    //          results are are unpredictable 
    ////////////////////////////////////////////////////////////////////////
	void imageToText();

protected:
    // image width
	unsigned widthM;
	// image height
	unsigned heightM;
	// maximum possible pixel value
	unsigned pixelMaxM;
	
	// mod of operation: It can be ios::in or ios::out
	ios_base::openmode modM;
	
    fstream* imgFileM;   // input image stream
    string imgNameM;    // image filename
	
	// image memory buffer
	byte* imgBufferM;
};

#endif //PGMIMAGE_H
