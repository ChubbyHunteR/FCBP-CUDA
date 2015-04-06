CC = nvcc
CFLAGS = -O3 -std=c++11

PROJECT = cbpc
OBJECTS = PGMImage.o PGMAverage.o PGMCBPCCUDA.o PredictorN.o PredictorNW.o PredictorGW.o PredictorW.o PredictorNE.o PredictorGN.o PredictorPL.o main.o
#HEADERS = config.h PGMImage.h PGMAverage.h PGMCBPCCUDA.h Predictor.h
#SOURCES = PGMImage.cpp PGMAverage.cpp PGMCBPCCUDA.cu Predictor.cu main.cpp
LIBRARIES =

all: $(PROJECT)

$(PROJECT): $(OBJECTS)
	$(CC) -o $(PROJECT) $(CFLAGS) $(OBJECTS)

PGMCBPCCUDA.o: PGMCBPCCUDA.cu PGMCBPCCUDA.h config.h
	$(CC) -c -o PGMCBPCCUDA.o $(CFLAGS) -c PGMCBPCCUDA.cu

PredictorN.o: predictors/PredictorN.cu predictors/PredictorN.h
	$(CC) -c -o PredictorN.o $(CFLAGS) -c predictors/PredictorN.cu

PredictorNW.o: predictors/PredictorNW.cu predictors/PredictorNW.h
	$(CC) -c -o PredictorNW.o $(CFLAGS) -c predictors/PredictorNW.cu

PredictorGW.o: predictors/PredictorGW.cu predictors/PredictorGW.h
	$(CC) -c -o PredictorGW.o $(CFLAGS) -c predictors/PredictorGW.cu

PredictorW.o: predictors/PredictorW.cu predictors/PredictorW.h
	$(CC) -c -o PredictorW.o $(CFLAGS) -c predictors/PredictorW.cu

PredictorNE.o: predictors/PredictorNE.cu predictors/PredictorNE.h
	$(CC) -c -o PredictorNE.o $(CFLAGS) -c predictors/PredictorNE.cu

PredictorGN.o: predictors/PredictorGN.cu predictors/PredictorGN.h
	$(CC) -c -o PredictorGN.o $(CFLAGS) -c predictors/PredictorGN.cu

PredictorPL.o: predictors/PredictorPL.cu predictors/PredictorPL.h
	$(CC) -c -o PredictorPL.o $(CFLAGS) -c predictors/PredictorPL.cu

PGMAverage.o: PGMAverage.cpp PGMAverage.h config.h
	$(CC) -c -o PGMAverage.o $(CFLAGS) PGMAverage.cpp

PGMImage.o: PGMImage.cpp PGMImage.h
	$(CC) -c -o PGMImage.o $(CFLAGS) -c PGMImage.cpp

main.o: main.cpp config.h
	$(CC) -c -o main.o $(CFLAGS) -c main.cpp

clean:
	-rm -f $(PROJECT) $(OBJECTS) *.o
