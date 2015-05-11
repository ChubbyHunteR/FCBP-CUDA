FCBP CUDA
=========

CBPC implementation in CUDA. FER undergraduate thesis, 2014/2015

This project depends on CUDA Toolkit (or the correct drivers, a CUDA capable graphics card and nvcc).

Release build:
make

Debug build:
make debug

Tester for the coder:
make tester

Clean:
make clean

Directory structure
===================
cbpc-cuda - CBP coder, CUDA.
cbpcTester - CBPC tester.
cbpd-cuda - CBP decoder, CUDA.
GRcodec - Golomb Rice coder and decoder.
PGM - Classes that handle PGM images and the 'error' file, which is a file that holds the difference between a prediction and the original.
staticPredictors-cuda - Static predictors that are weighted before they produce a final prediction, CUDA.
