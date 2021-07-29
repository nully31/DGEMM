CC=gcc
ifndef avx512
CFLAGS=-std=c11 -g -mavx2 -fopenmp -O3 -lblas -lcblas
EXC:=$(wildcard dgemm_avx512*)
SRCS:=$(filter-out $(EXC), $(wildcard *.c))
else
CFLAGS=-std=c11 -g -mavx2 -fopenmp -O3 -lblas -lcblas -mavx512f
SRCS:=$(wildcard *.c)
endif
OBJS:=$(SRCS:.c=.o)


main: $(OBJS)
	$(CC) -o main $(OBJS) $(CFLAGS)

$(OBJS): dgemm.h

run: main
	./main $(size)

clean:
	rm -f main *.o *~ tmp*

.PHONY: run clean avx512