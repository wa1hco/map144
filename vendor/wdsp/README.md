# vendor/wdsp/

Vendored subset of Warren Pratt's (NR0V) WDSP library — just the wideband
noise blanker (`nob.c` / `nob.h`) plus a thin Linux stub header and a
MAP144-local shim.  See [SOURCE.md](SOURCE.md) for upstream provenance.

## Contents

| File          | Origin   | Role |
|---------------|----------|------|
| `nob.c`       | upstream | WDSP wideband noise blanker, verbatim |
| `nob.h`       | upstream | WDSP API, verbatim |
| `comm.h`      | MAP144   | Cross-platform stub of upstream `comm.h`: native Windows APIs on `_WIN32`, POSIX/ELF replacements elsewhere |
| `nob_shim.c`  | MAP144   | One-function helper (`xanb_buf`) for buffer-size flexibility without state reset |
| `Makefile`    | MAP144   | Build `libnob.so` (Linux/macOS) or `libnob.dll` (Windows/MinGW) |
| `SOURCE.md`   | MAP144   | Upstream commit hash + re-sync instructions |

## Build

### Linux / macOS

```
make -C vendor/wdsp
```

Produces `libnob.so` in the same directory.

### Windows

`nob.c` is upstream *Windows* code, so `comm.h` uses the native Win32 APIs
(`windows.h` `CRITICAL_SECTION`, `malloc.h` `_aligned_malloc`, real
`__declspec(dllexport)`) under `_WIN32`.  Build with **either** toolchain on a
machine that has it; drop the resulting `libnob.dll` into `vendor\wdsp\`.

MinGW-w64 (gcc) — via the Makefile:
```
mingw32-make -C vendor\wdsp
```
or directly:
```
gcc -O2 -std=gnu99 -shared -o vendor\wdsp\libnob.dll vendor\wdsp\nob.c vendor\wdsp\nob_shim.c
```

MSVC (Developer Command Prompt):
```
cl /O2 /LD /Fe:vendor\wdsp\libnob.dll vendor\wdsp\nob.c vendor\wdsp\nob_shim.c
```

The Python wrapper [`map144_app/nb_nr0v_wideband.py`](../../map144_app/nb_nr0v_wideband.py)
resolves the platform library (`libnob.dll` / `.dylib` / `.so`) relative to the
repo root; once present, the "NR0V-Wideband" backend becomes selectable.

Clean (Linux/macOS):

```
make -C vendor/wdsp clean
```

## Dependencies

- A C99 compiler: GCC/Clang (Linux/macOS) or MinGW-w64 gcc / MSVC (Windows).
- Linux/macOS: `libpthread`, `libm` (both standard).
- Windows: Win32 + CRT only (`windows.h`, `malloc.h`) — no pthread, no fftw3,
  no rest-of-WDSP.

## Why only nob.c?

`nob.c` is Warren's **wideband** (time-domain state-machine) noise blanker.
The separate spectral/LPC blanker (`snb.c`) is planned as a later stage —
see [docs/noise_blanker_plan.md](../../docs/noise_blanker_plan.md).  Keeping
the Stage-2 vendor surface minimal reduces integration risk and makes
licensing/provenance trivial to audit.
