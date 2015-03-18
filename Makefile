CC = nvcc
CFLAGS = -O3 -std=c++11

PROJECT = cbpc
OBJECTS = PGMImage.o PGMAverage.o PGMAverageCUDA.o main.o
HEADERS = config.h PGMImage.h PGMAverage.h PGMAverageCUDA.h
SOURCES = PGMImage.cpp PGMAverage.cpp PGMAverageCUDA.cu main.cpp
LIBRARIES =

$(PROJECT): $(OBJECTS)
	$(CC) -o $(PROJECT) $(CFLAGS) $(OBJECTS)

PGMAverageCUDA.o: PGMAverageCUDA.cu PGMAverageCUDA.h config.h
	$(CC) -c -o PGMAverageCUDA.o $(CFLAGS) -c PGMAverageCUDA.cu

PGMAverage.o: PGMAverage.cpp PGMAverage.h config.h
	$(CC) -c -o PGMAverage.o $(CFLAGS) PGMAverage.cpp

PGMImage.o: PGMImage.cpp PGMImage.h
	$(CC) -c -o PGMImage.o $(CFLAGS) -c PGMImage.cpp

main.o: main.cpp config.h
	$(CC) -c -o main.o $(CFLAGS) -c main.cpp

clean:
	-rm -f $(PROJECT) $(OBJECTS) *.o
