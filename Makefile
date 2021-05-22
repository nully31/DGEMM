CC=gcc
CFLAGS=-std=c11 -g -mavx2 -mavx512f -fopenmp -O3
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

main: $(OBJS)
	$(CC) -o main $(OBJS) $(CFLAGS)

$(OBJS): dgemm.h

test:
	./main $(size)

clean:
	rm -f main *.o *~ tmp*

.PHONY: clean