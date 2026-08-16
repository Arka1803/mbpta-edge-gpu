CC=gcc
CFLAGS=-Wall -O2

all: profiler

profiler: profiler.c
	$(CC) $(CFLAGS) -o profiler profiler.c

clean:
	rm -f profiler
