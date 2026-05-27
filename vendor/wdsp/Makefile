# Build libnob.so — a standalone shared library containing WDSP's
# wideband noise blanker (nob.c), exported for ctypes use from Python.
#
# Only nob.c is compiled. comm.h here is a MAP144 stub that provides
# just the symbols nob.c references, not the full WDSP header stack.

CC       ?= cc
CFLAGS   ?= -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
            -fPIC -fvisibility=hidden -std=gnu99
LDFLAGS  ?= -shared
LIBS      = -lpthread -lm

TARGET    = libnob.so
OBJS      = nob.o nob_shim.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

nob.o: nob.c comm.h nob.h
	$(CC) $(CFLAGS) -c nob.c -o nob.o

nob_shim.o: nob_shim.c comm.h nob.h
	$(CC) $(CFLAGS) -c nob_shim.c -o nob_shim.o

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
