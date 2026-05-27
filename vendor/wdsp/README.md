# vendor/wdsp/

Vendored subset of Warren Pratt's (NR0V) WDSP library — just the wideband
noise blanker (`nob.c` / `nob.h`) plus a thin Linux stub header and a
MAP144-local shim.  See [SOURCE.md](SOURCE.md) for upstream provenance.

## Contents

| File          | Origin   | Role |
|---------------|----------|------|
| `nob.c`       | upstream | WDSP wideband noise blanker, verbatim |
| `nob.h`       | upstream | WDSP API, verbatim |
| `comm.h`      | MAP144   | POSIX/ELF replacement for upstream `comm.h` mega-header |
| `nob_shim.c`  | MAP144   | One-function helper (`xanb_buf`) for buffer-size flexibility without state reset |
| `Makefile`    | MAP144   | Build `libnob.so` |
| `SOURCE.md`   | MAP144   | Upstream commit hash + re-sync instructions |

## Build

```
make -C vendor/wdsp
```

Produces `libnob.so` in the same directory.  The Python wrapper
[`map144_app/nb_nr0v_wideband.py`](../../map144_app/nb_nr0v_wideband.py)
resolves it relative to the repo root.

Clean:

```
make -C vendor/wdsp clean
```

## Dependencies

- `cc` with `-std=gnu99` support (GCC or Clang).
- `libpthread`, `libm` (both standard).
- No fftw3, no Windows headers, no rest-of-WDSP.

## Why only nob.c?

`nob.c` is Warren's **wideband** (time-domain state-machine) noise blanker.
The separate spectral/LPC blanker (`snb.c`) is planned as a later stage —
see [docs/noise_blanker_plan.md](../../docs/noise_blanker_plan.md).  Keeping
the Stage-2 vendor surface minimal reduces integration risk and makes
licensing/provenance trivial to audit.
