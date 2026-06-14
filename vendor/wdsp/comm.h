/*  comm.h — MAP144 stub of WDSP comm.h for nob.c only.
 *
 *  Upstream comm.h pulls in every WDSP subsystem plus Windows.h, fftw3,
 *  and dozens of unrelated headers.  For the noise-blanker we only need:
 *    - math / memory helpers (PI, sqrt, cos, exp, memset, memcpy)
 *    - malloc0 / _aligned_free — WDSP's aligned-allocator pair
 *    - CRITICAL_SECTION family — mutex primitives
 *    - PORT / __declspec(dllexport) — DLL export markers
 *    - complex typedef — interleaved double pair (NOT C99 _Complex)
 *
 *  nob.c is upstream *Windows* code, so on Windows (_WIN32) we let it use
 *  the native APIs it was written against (windows.h CRITICAL_SECTION,
 *  malloc.h _aligned_malloc, real __declspec(dllexport)).  On Linux/POSIX
 *  we provide drop-in replacements so the same nob.c compiles unchanged.
 *
 *  No other WDSP files are compiled, so unrelated headers are not pulled
 *  in here.
 */
#ifndef _map144_wdsp_comm_h
#define _map144_wdsp_comm_h

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef PI
#define PI 3.1415926535897932
#endif

/* WDSP's complex type is an interleaved double pair, NOT C99 _Complex. */
typedef double complex[2];

#ifdef _WIN32
/* ── Windows: nob.c's native environment ─────────────────────────────────
 * CRITICAL_SECTION + Initialize/Delete/Enter/Leave come from <windows.h>;
 * _aligned_malloc / _aligned_free from <malloc.h>; __declspec(dllexport)
 * is the real thing, so we must NOT redefine it. */
#include <windows.h>
#include <malloc.h>
#define PORT __declspec(dllexport)

#else
/* ── Linux/POSIX replacements so nob.c compiles unchanged ────────────────
 * PORT / __declspec become default-visibility exports on ELF. */
#include <pthread.h>
#define PORT           __attribute__((visibility("default")))
#define __declspec(x)  __attribute__((visibility("default")))

/* Windows CRITICAL_SECTION → pthread_mutex_t (one CS per ANB instance). */
typedef pthread_mutex_t CRITICAL_SECTION;

static inline void InitializeCriticalSectionAndSpinCount(CRITICAL_SECTION *cs, unsigned long spin) {
    (void)spin;
    pthread_mutex_init(cs, NULL);
}
static inline void DeleteCriticalSection(CRITICAL_SECTION *cs) { pthread_mutex_destroy(cs); }
static inline void EnterCriticalSection (CRITICAL_SECTION *cs) { pthread_mutex_lock(cs); }
static inline void LeaveCriticalSection (CRITICAL_SECTION *cs) { pthread_mutex_unlock(cs); }

/* WDSP aligned-allocator pair (Windows-native on _WIN32, stubbed here). */
static inline void *_aligned_malloc(size_t size, size_t alignment) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}
static inline void _aligned_free(void *p) { free(p); }
#endif /* _WIN32 */

/* malloc0: zero-initialized, 16-byte-aligned.  Same semantics as WDSP's.
 * (_aligned_malloc is native on Windows via <malloc.h>, stubbed above on POSIX.) */
static inline void *malloc0(int size) {
    void *p = _aligned_malloc((size_t)size, 16);
    if (p != NULL) memset(p, 0, (size_t)size);
    return p;
}

/* -- nob.h forward ------------------------------------------------------- */
#include "nob.h"

#endif /* _map144_wdsp_comm_h */
