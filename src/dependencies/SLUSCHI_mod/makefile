
CC = ifort

SRC = $(wildcard *.f90)
TAR = $(SRC:.f90=.x)

.PHONY: all clean

all: $(TAR)

%.x: %.f90
	$(CC) -o $@ $<

clean:
	rm -f $(TAR)

