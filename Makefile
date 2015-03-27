CC = nvcc
CFLAGS = -O3 -std=c++11

PROJECT = cbpc
OBJECTS = PGMImage.o PGMAverage.o PGMCBPCCUDA.o Predictor.o main.o
HEADERS = config.h PGMImage.h PGMAverage.h PGMCBPCCUDA.h Predictor.h
SOURCES = PGMImage.cpp PGMAverage.cpp PGMCBPCCUDA.cu Predictor.cpp main.cpp
LIBRARIES =

$(PROJECT): $(OBJECTS)
	$(CC) -o $(PROJECT) $(CFLAGS) $(OBJECTS)

PGMCBPCCUDA.o: PGMCBPCCUDA.cu PGMCBPCCUDA.h config.h
	$(CC) -c -o PGMCBPCCUDA.o $(CFLAGS) -c PGMCBPCCUDA.cu

Predictor.o: predictors/Predictor.cpp predictors/Predictor.h
	$(CC) -c -o Predictor.o $(CFLAGS) -c predictors/Predictor.cpp

PGMAverage.o: PGMAverage.cpp PGMAverage.h config.h
	$(CC) -c -o PGMAverage.o $(CFLAGS) PGMAverage.cpp

PGMImage.o: PGMImage.cpp PGMImage.h
	$(CC) -c -o PGMImage.o $(CFLAGS) -c PGMImage.cpp

main.o: main.cpp config.h
	$(CC) -c -o main.o $(CFLAGS) -c main.cpp

clean:
	-rm -f $(PROJECT) $(OBJECTS) *.o
