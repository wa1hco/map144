# WDSP source provenance

## Files

`nob.c` and `nob.h` in this directory are copied **byte-for-byte verbatim** from
Warren Pratt (NR0V)'s WDSP library, as embedded in Thetis.

## Upstream

- Repository: https://github.com/TAPR/OpenHPSDR-Thetis
- Subdirectory: `Project Files/Source/wdsp/`
- Commit: `619b13f3d57b4e2261974db859d6dcf14b455214`
- Files imported: `nob.c` (448 lines), `nob.h` (124 lines)

## License

Both files carry Warren Pratt's original copyright headers:

> Copyright (C) 2013, 2014 Warren Pratt, NR0V
> This program is free software; you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation; either version 2 of the License, or
> (at your option) any later version.

MAP144 is GPL-3. GPL-2-or-later is compatible with GPL-3, so vendoring these
files here under MAP144's GPL-3 license is compliant.

## MAP144 additions

The following files in this directory are **not** from upstream and are MAP144
code:

- `comm.h` — a minimal stub of WDSP's mega-header, providing only the symbols
  `nob.c` references (malloc0, CRITICAL_SECTION family, complex typedef, PI,
  PORT macro) as POSIX/ELF equivalents. Upstream `comm.h` pulls in Windows.h,
  fftw3, and every WDSP subsystem — none of which are needed for the blanker.
- `nob_shim.c` — one-function helper (`xanb_buf`) that lets the caller change
  buffer pointers and size between invocations without calling the upstream
  `setSize_anb`, which would reset state.
- `Makefile` — build `libnob.so` from `nob.c` + `nob_shim.c`.

## Re-syncing with upstream

To update the vendored WDSP sources:

1. Clone the upstream at a newer commit.
2. Overwrite `nob.c` and `nob.h` here from `Project Files/Source/wdsp/`.
3. Update the commit hash above.
4. Review the diff — if upstream added new external symbols (Windows API,
   other WDSP modules), extend `comm.h` accordingly.
5. `make clean && make` and run the MAP144 blanker self-test.
