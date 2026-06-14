# Build the WDSP wideband noise blanker (nob.c + nob_shim.c) as a standalone
# shared library, exported for ctypes use from map144_app/nb_nr0v_wideband.py.
#
# Only nob.c (+ the MAP144 nob_shim.c) is compiled; comm.h is a MAP144 stub
# providing just the symbols nob.c references, not the full WDSP header stack.
#
#   Linux/macOS : make            -> libnob.so (.dylib on macOS via CC)
#   Windows     : mingw32-make    -> libnob.dll   (MinGW-w64 gcc)
#                 (MSVC: see vendor/wdsp/README.md for the cl.exe one-liner)

OBJS = nob.o nob_shim.o

ifeq ($(OS),Windows_NT)
  # MinGW-w64.  nob.c is upstream Windows code; __declspec(dllexport) in
  # nob.h/comm.h exports the API, so no -fPIC/-fvisibility and no pthread/m.
  CC      ?= gcc
  CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -std=gnu99
  LDFLAGS ?= -shared
  LIBS     =
  TARGET   = libnob.dll
  RM       = del /q
else
  CC      ?= cc
  CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
             -fPIC -fvisibility=hidden -std=gnu99
  LDFLAGS ?= -shared
  LIBS     = -lpthread -lm
  TARGET   = libnob.so
  RM       = rm -f
endif

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

nob.o: nob.c comm.h nob.h
	$(CC) $(CFLAGS) -c nob.c -o nob.o

nob_shim.o: nob_shim.c comm.h nob.h
	$(CC) $(CFLAGS) -c nob_shim.c -o nob_shim.o

clean:
	$(RM) $(OBJS) $(TARGET)

.PHONY: all clean
