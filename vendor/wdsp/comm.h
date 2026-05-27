/*  comm.h — MAP144 stub of WDSP comm.h for nob.c only.
 *
 *  Upstream comm.h pulls in every WDSP subsystem plus Windows.h, fftw3,
 *  and dozens of unrelated headers.  For the noise-blanker we only need:
 *    - math / memory helpers (PI, sqrt, cos, exp, memset, memcpy)
 *    - malloc0 / _aligned_free — WDSP's aligned-allocator pair
 *    - CRITICAL_SECTION family — Windows mutex primitives
 *    - PORT / __declspec(dllexport) — Windows DLL export markers
 *    - complex typedef — interleaved double pair (NOT C99 _Complex)
 *
 *  This header provides Linux/POSIX replacements so nob.c compiles
 *  unchanged.  The Windows mutex API is stubbed to pthread_mutex_t —
 *  lock ordering matches the original (one CS per ANB instance).
 *
 *  No other WDSP files are compiled, so unrelated headers are not
 *  pulled in here.
 */
#ifndef _map144_wdsp_comm_h
#define _map144_wdsp_comm_h

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>

#ifndef PI
#define PI 3.1415926535897932
#endif

/* PORT / __declspec(dllexport) become default-visibility exports on ELF. */
#define PORT       __attribute__((visibility("default")))
#define __declspec(x)  __attribute__((visibility("default")))

/* WDSP's complex type is an interleaved double pair, NOT C99 _Complex. */
typedef double complex[2];

/* -- Windows CRITICAL_SECTION → pthread_mutex_t --------------------------- */
typedef pthread_mutex_t CRITICAL_SECTION;

static inline void InitializeCriticalSectionAndSpinCount(CRITICAL_SECTION *cs, unsigned long spin) {
    (void)spin;
    pthread_mutex_init(cs, NULL);
}
static inline void DeleteCriticalSection(CRITICAL_SECTION *cs) { pthread_mutex_destroy(cs); }
static inline void EnterCriticalSection (CRITICAL_SECTION *cs) { pthread_mutex_lock(cs); }
static inline void LeaveCriticalSection (CRITICAL_SECTION *cs) { pthread_mutex_unlock(cs); }

/* -- WDSP aligned-allocator pair ----------------------------------------- */
static inline void *_aligned_malloc(size_t size, size_t alignment) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}
static inline void _aligned_free(void *p) { free(p); }

/* malloc0: zero-initialized, 16-byte-aligned.  Same semantics as WDSP's. */
static inline void *malloc0(int size) {
    void *p = _aligned_malloc((size_t)size, 16);
    if (p != NULL) memset(p, 0, (size_t)size);
    return p;
}

/* -- nob.h forward ------------------------------------------------------- */
#include "nob.h"

#endif /* _map144_wdsp_comm_h */
