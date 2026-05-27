/*  nob_shim.c — thin MAP144 helper on top of WDSP nob.c.
 *
 *  Upstream's xanb(ANB) uses a->buffsize as the loop bound, and the
 *  public setSize_anb() also calls initBlanker() which resets the state
 *  machine and ring buffer — we cannot use it between chunks without
 *  losing continuity.  This shim exposes a call that updates in/out
 *  pointers and buffsize in-place (no state reset), then invokes xanb.
 *
 *  Kept here in a separate translation unit so nob.c stays byte-for-byte
 *  identical to the upstream file in vendor/wdsp/.
 */
#include "comm.h"

extern void xanb (ANB a);

PORT
void xanb_buf(ANB a, double *in_buf, double *out_buf, int n)
{
    a->in       = in_buf;
    a->out      = out_buf;
    a->buffsize = n;
    xanb(a);
}
