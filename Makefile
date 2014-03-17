CC=g++-4.9

all:
	${CC} -o main main.cpp scvb.cpp -O3 -fopenmp -msse4 -std=c++11
