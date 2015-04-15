CC = nvcc
CFLAGS = -O3 -std=c++11
CFLAGS_DEBUG = -g -G -std=c++11 -DDEBUG

PROJECT = cbpc
OBJECTS = PGMImage.o PGMAverage.o PGMCBPCCUDA.o PredictorN.o PredictorNW.o PredictorGW.o PredictorW.o PredictorNE.o PredictorGN.o PredictorPL.o main.o
OBJECTS_DEBUG = PGMImage_d.o PGMAverage_d.o PGMCBPCCUDA_d.o PredictorN_d.o PredictorNW_d.o PredictorGW_d.o PredictorW_d.o PredictorNE_d.o PredictorGN_d.o PredictorPL_d.o main_d.o


###########
# RELEASE #
###########

all: $(PROJECT)

$(PROJECT): $(OBJECTS)
	$(CC) -o $(PROJECT) $(CFLAGS) $(OBJECTS)

PGMCBPCCUDA.o: PGMCBPCCUDA.cu PGMCBPCCUDA.h config.h util.h
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


#########
# DEBUG #
#########
debug: $(OBJECTS_DEBUG)
	$(CC) -o $(PROJECT) $(CFLAGS_DEBUG) $(OBJECTS_DEBUG)

PGMCBPCCUDA_d.o: PGMCBPCCUDA.cu PGMCBPCCUDA.h config.h util.h
	$(CC) -c -o PGMCBPCCUDA_d.o $(CFLAGS_DEBUG) -c PGMCBPCCUDA.cu

PredictorN_d.o: predictors/PredictorN.cu predictors/PredictorN.h
	$(CC) -c -o PredictorN_d.o $(CFLAGS_DEBUG) -c predictors/PredictorN.cu

PredictorNW_d.o: predictors/PredictorNW.cu predictors/PredictorNW.h
	$(CC) -c -o PredictorNW_d.o $(CFLAGS_DEBUG) -c predictors/PredictorNW.cu

PredictorGW_d.o: predictors/PredictorGW.cu predictors/PredictorGW.h
	$(CC) -c -o PredictorGW_d.o $(CFLAGS_DEBUG) -c predictors/PredictorGW.cu

PredictorW_d.o: predictors/PredictorW.cu predictors/PredictorW.h
	$(CC) -c -o PredictorW_d.o $(CFLAGS_DEBUG) -c predictors/PredictorW.cu

PredictorNE_d.o: predictors/PredictorNE.cu predictors/PredictorNE.h
	$(CC) -c -o PredictorNE_d.o $(CFLAGS_DEBUG) -c predictors/PredictorNE.cu

PredictorGN_d.o: predictors/PredictorGN.cu predictors/PredictorGN.h
	$(CC) -c -o PredictorGN_d.o $(CFLAGS_DEBUG) -c predictors/PredictorGN.cu

PredictorPL_d.o: predictors/PredictorPL.cu predictors/PredictorPL.h
	$(CC) -c -o PredictorPL_d.o $(CFLAGS_DEBUG) -c predictors/PredictorPL.cu

PGMAverage_d.o: PGMAverage.cpp PGMAverage.h config.h
	$(CC) -c -o PGMAverage_d.o $(CFLAGS_DEBUG) PGMAverage.cpp

PGMImage_d.o: PGMImage.cpp PGMImage.h
	$(CC) -c -o PGMImage_d.o $(CFLAGS_DEBUG) -c PGMImage.cpp

main_d.o: main.cpp config.h
	$(CC) -c -o main_d.o $(CFLAGS_DEBUG) -c main.cpp



clean:
	-rm -f $(PROJECT) *.o
