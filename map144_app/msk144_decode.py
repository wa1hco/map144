"""msk144_decode.py — Pure-Python MSK144 frame decoder.

Implements the full soft-decision decode chain:
    phase correction → matched filter → soft bits →
    belief-propagation LDPC decode → message unpack

Replaces the jt9 subprocess call in the SPD path, so the soft-decision
decoder operates directly on the SPD-aligned frame without re-running
jt9's internal detection gate.

References
----------
msk144decodeframe.f90   — phase correction, matched filter, soft bits
bpdecode128_90.f90      — log-domain belief propagation
ldpc_128_90_reordered_parity.f90 — H-matrix (Mn / Nm / nrw)
chkcrc13a.f90 + crc13.cpp — CRC-13 check (polynomial 0x15D7)
packjt77.f90            — message unpacking (unpack28, to_grid4, unpack77)
"""

import numpy as np
from typing import Optional

try:
    import numba
    _NUMBA = True
    import logging as _logging
    _logging.getLogger(__name__).info("numba %s JIT enabled for BP decoder", numba.__version__)
except ImportError:
    _NUMBA = False
    import logging as _logging
    _logging.getLogger(__name__).warning("numba not found — BP decoder using slow pure-Python path")

# ── Constants ─────────────────────────────────────────────────────────────────

NSPM = 864          # samples per MSK144 frame at 12 kHz
FS   = 12000.0

# ── LDPC (128, 90) matrix ─────────────────────────────────────────────────────
# Translated directly from ldpc_128_90_reordered_parity.f90.
# All values converted to 0-based indexing.

# Mn[n, 0..2] — the 3 check-node indices for variable node n  (N=128 nodes)
_MN_1IDX = [
    [21,34,36],[1,8,28],[2,9,37],[3,7,19],[4,16,32],[2,5,22],[6,13,25],
    [10,31,33],[11,24,27],[12,15,23],[14,18,26],[17,20,29],[17,30,34],
    [6,34,35],[1,10,30],[3,18,23],[4,12,25],[5,28,36],[7,14,21],[8,15,31],
    [9,27,32],[11,19,35],[13,16,37],[20,24,38],[21,22,26],[12,29,33],
    [1,17,35],[2,28,30],[3,10,32],[4,8,36],[5,19,29],[6,20,27],[7,22,37],
    [9,11,33],[13,24,26],[14,31,34],[15,16,25],[13,18,38],[8,20,23],
    [1,32,33],[2,17,19],[3,24,34],[4,7,38],[5,11,31],[6,18,21],[9,15,36],
    [10,16,28],[12,26,30],[14,27,29],[22,25,35],[23,30,32],[4,11,37],
    [1,14,23],[2,8,25],[3,13,27],[5,10,37],[6,16,31],[7,15,18],[9,22,24],
    [12,19,36],[17,26,38],[20,21,33],[20,28,35],[4,29,34],[1,26,36],
    [2,23,34],[3,9,38],[5,6,17],[7,27,35],[8,14,32],[10,15,22],[11,18,29],
    [12,13,28],[16,19,33],[21,25,31],[24,30,37],[1,3,21],[2,18,31],
    [4,6,9],[5,8,33],[7,29,32],[10,13,19],[11,22,23],[12,27,34],[14,15,30],
    [16,27,38],[17,28,37],[20,25,26],[5,24,35],[3,6,36],[1,12,31],
    [2,4,33],[3,16,30],[1,2,24],[5,23,27],[6,28,32],[7,17,36],[8,22,38],
    [9,18,20],[10,21,29],[11,13,34],[4,14,20],[11,30,38],[14,35,37],
    [15,19,26],[3,28,29],[7,8,9],[5,18,34],[13,15,17],[12,16,35],
    [10,23,25],[19,21,37],[17,27,31],[24,25,36],[1,18,19],[6,26,33],
    [22,31,32],[3,20,22],[4,21,27],[2,13,29],[6,7,12],[15,24,32],
    [9,25,30],[23,37,38],[5,16,26],[11,14,28],[33,36,38],[8,10,35],
]
# Convert to 0-indexed numpy array
Mn = np.array(_MN_1IDX, dtype=np.int32) - 1   # shape (128, 3)

# Nm[m, 0..nrw[m]-1] — variable node indices for check node m  (M=38 nodes)
# 0 entries in Fortran become -1 here (unused slots)
_NM_1IDX = [
    [2,15,27,40,53,65,77,91,94,115,0],
    [3,6,28,41,54,66,78,92,94,120,0],
    [4,16,29,42,55,67,77,90,93,106,118],
    [5,17,30,43,52,64,79,92,102,119,0],
    [6,18,31,44,56,68,80,89,95,108,125],
    [7,14,32,45,57,68,79,90,96,116,121],
    [4,19,33,43,58,69,81,97,107,121,0],
    [2,20,30,39,54,70,80,98,107,128,0],
    [3,21,34,46,59,67,79,99,107,123,0],
    [8,15,29,47,56,71,82,100,111,128,0],
    [9,22,34,44,52,72,83,101,103,126,0],
    [10,17,26,48,60,73,84,91,110,121,0],
    [7,23,35,38,55,73,82,101,109,120,0],
    [11,19,36,49,53,70,85,102,104,126,0],
    [10,20,37,46,58,71,85,105,109,122,0],
    [5,23,37,47,57,74,86,93,110,125,0],
    [12,13,27,41,61,68,87,97,109,113,0],
    [11,16,38,45,58,72,78,99,108,115,0],
    [4,22,31,41,60,74,82,105,112,115,0],
    [12,24,32,39,62,63,88,99,102,118,0],
    [1,19,25,45,62,75,77,100,112,119,0],
    [6,25,33,50,59,71,83,98,117,118,0],
    [10,16,39,51,53,66,83,95,111,124,0],
    [9,24,35,42,59,76,89,94,114,122,0],
    [7,17,37,50,54,75,88,111,114,123,0],
    [11,25,35,48,61,65,88,105,116,125,0],
    [9,21,32,49,55,69,84,86,95,113,119],
    [2,18,28,47,63,73,87,96,106,126,0],
    [12,26,31,49,64,72,81,100,106,120,0],
    [13,15,28,48,51,76,85,93,103,123,0],
    [8,20,36,44,57,75,78,91,113,117,0],
    [5,21,29,40,51,70,81,96,117,122,0],
    [8,26,34,40,62,74,80,92,116,127,0],
    [1,13,14,36,42,64,66,84,101,108,0],
    [14,22,27,50,63,69,89,104,110,128,0],
    [1,18,30,46,60,65,90,97,114,127,0],
    [3,23,33,52,56,76,87,104,112,124,0],
    [24,38,43,61,67,86,98,103,124,127,0],
]
_NRW_1IDX = [
    10,10,11,10,11,11,10,10,10,10,10,10,10,10,10,10,10,10,
    10,10,10,10,10,10,10,10,11,10,10,10,10,10,10,10,10,10,
    10,10,
]
nrw = np.array(_NRW_1IDX, dtype=np.int32)  # shape (38,)

# Build 0-indexed Nm: shape (38, 11), -1 for unused slots
Nm = np.full((38, 11), -1, dtype=np.int32)
for m, row in enumerate(_NM_1IDX):
    for k, v in enumerate(row):
        if v != 0:
            Nm[m, k] = v - 1

N_LDPC = 128
K_LDPC = 90
M_LDPC = 38   # N - K

# ── CRC-13 (polynomial 0x15D7) ────────────────────────────────────────────────

def _crc13(data: bytes) -> int:
    """CRC-13 matching Boost augmented_crc<13, 0x15D7> (MSB-first, no reflection).

    Boost augmented_crc shifts the input bit into the LSB of the register BEFORE
    XOR'ing with the polynomial (quotient-first form), unlike the standard CRC
    which XORs input with the MSB before shifting.
    """
    POLY = 0x15D7
    crc = 0
    for byte in data:
        for bit in range(7, -1, -1):
            quotient = (crc >> 12) & 1          # MSB of register
            crc = ((crc << 1) | ((byte >> bit) & 1)) & 0x1FFF
            if quotient:
                crc ^= POLY
    return crc


def _check_crc13(decoded_90: np.ndarray) -> bool:
    """Verify CRC-13 of 90-bit decoded word (77 message bits + 13 CRC bits).

    Mirrors chkcrc13a.f90: zero the CRC bits, pack into 12 bytes,
    compute CRC13, compare with received CRC.
    """
    # Received CRC: bits 77..89 (0-indexed), MSB first
    ncrc13 = 0
    for b in decoded_90[77:90]:
        ncrc13 = (ncrc13 << 1) | int(b)

    # Pack 90 bits into 12 bytes with CRC bits zeroed
    buf = bytearray(12)
    for i in range(77):                 # message bits only
        buf[i >> 3] |= int(decoded_90[i]) << (7 - (i & 7))
    # bytes beyond bit 76 remain 0 (CRC zeroed)

    icrc13 = _crc13(bytes(buf))
    return ncrc13 == icrc13


# ── Belief-propagation LDPC decoder ──────────────────────────────────────────

def _platanh(x: float) -> float:
    """Piecewise-linear atanh approximation (platanh.f90)."""
    z = abs(x)
    sign = 1.0 if x >= 0 else -1.0
    if z <= 0.664:
        return x / 0.83
    elif z <= 0.9217:
        return sign * (z - 0.4064) / 0.322
    elif z <= 0.9951:
        return sign * (z - 0.8378) / 0.0524
    elif z <= 0.9998:
        return sign * (z - 0.9914) / 0.0012
    else:
        return sign * 7.0


if _NUMBA:
    @numba.njit(cache=True)
    def _platanh_nb(x: float) -> float:
        z = abs(x)
        sign = 1.0 if x >= 0.0 else -1.0
        if z <= 0.664:
            return x / 0.83
        elif z <= 0.9217:
            return sign * (z - 0.4064) / 0.322
        elif z <= 0.9951:
            return sign * (z - 0.8378) / 0.0524
        elif z <= 0.9998:
            return sign * (z - 0.9914) / 0.0012
        else:
            return sign * 7.0

    @numba.njit(cache=True)
    def _bp_inner(llr, Mn, Nm, nrw, max_iterations):
        """Numba-compiled BP inner loop.

        Returns (cw_90, nharderror) where nharderror == -1 means failure.
        When nharderror >= 0 the caller must verify CRC on cw_90[:77].
        """
        N, M, K = 128, 38, 90
        tov = np.zeros((3, N), dtype=np.float32)
        toc = np.zeros((11, M), dtype=np.float32)

        for j in range(M):
            for i in range(nrw[j]):
                toc[i, j] = llr[Nm[j, i]]

        nclast = 0
        ncnt   = 0

        for iteration in range(max_iterations + 1):
            zn = llr + tov[0] + tov[1] + tov[2]

            cw = np.empty(N, dtype=np.int8)
            for i in range(N):
                cw[i] = np.int8(1) if zn[i] > 0.0 else np.int8(0)

            ncheck = 0
            for j in range(M):
                s = 0
                for ii in range(nrw[j]):
                    s += int(cw[Nm[j, ii]])
                if s % 2 != 0:
                    ncheck += 1

            if ncheck == 0:
                nharderror = 0
                for ii in range(N):
                    if (2 * int(cw[ii]) - 1) * llr[ii] < 0.0:
                        nharderror += 1
                return cw[:K].copy(), nharderror

            if iteration > 0:
                nd = ncheck - nclast
                if nd >= 0:
                    ncnt += 1
                else:
                    ncnt = 0
                if ncnt >= 3 and iteration >= 5 and ncheck > 10:
                    return cw[:K].copy(), -1
            nclast = ncheck

            for j in range(M):
                for i in range(nrw[j]):
                    ibj = Nm[j, i]
                    val = zn[ibj]
                    for kk in range(3):
                        if Mn[ibj, kk] == j:
                            val -= tov[kk, ibj]
                    toc[i, j] = val

            tanhtoc = np.tanh(-toc * 0.5)

            for i in range(N):
                for kk in range(3):
                    ichk = Mn[i, kk]
                    prod = 1.0
                    for ii in range(nrw[ichk]):
                        if Nm[ichk, ii] != i:
                            prod *= tanhtoc[ii, ichk]
                    tov[kk, i] = 2.0 * _platanh_nb(-prod)

        return cw[:K].copy(), -1


def _bpdecode128_90(
    llr: np.ndarray,
    max_iterations: int = 10,
) -> tuple[Optional[np.ndarray], int]:
    """Log-domain belief propagation for the (128, 90) LDPC code.

    Parameters
    ----------
    llr            : float32 array shape (128,) — channel log-likelihood ratios
    max_iterations : int — maximum BP iterations (default 10, matches Fortran)

    Returns
    -------
    message77 : int8 array shape (77,) or None on failure
    nharderror: int ≥ 0 on success, −1 on failure
    """
    if _NUMBA:
        cw90, nharderror = _bp_inner(llr, Mn, Nm, nrw, max_iterations)
        if nharderror == -1:
            return None, -1
        if _check_crc13(cw90):
            return cw90[:77].copy(), nharderror
        return None, -1

    # ── Pure-Python fallback (no numba) ──────────────────────────────────────
    N, M = N_LDPC, M_LDPC

    tov = np.zeros((3, N), dtype=np.float32)
    toc = np.zeros((11, M), dtype=np.float32)

    for j in range(M):
        for i in range(nrw[j]):
            toc[i, j] = llr[Nm[j, i]]

    nclast = 0
    ncnt   = 0

    for iteration in range(max_iterations + 1):

        zn = llr + tov.sum(axis=0)
        cw = (zn > 0).astype(np.int8)

        ncheck = 0
        for j in range(M):
            s = int(cw[Nm[j, :nrw[j]]].sum()) % 2
            if s:
                ncheck += 1

        if ncheck == 0:
            decoded = cw[:K_LDPC]
            if _check_crc13(decoded):
                nharderror = int((((2 * cw.astype(np.float32) - 1) * llr) < 0).sum())
                return decoded[:77].copy(), nharderror

        if iteration > 0:
            nd = ncheck - nclast
            if nd >= 0:
                ncnt += 1
            else:
                ncnt = 0
            if ncnt >= 3 and iteration >= 5 and ncheck > 10:
                return None, -1
        nclast = ncheck

        for j in range(M):
            for i in range(nrw[j]):
                ibj = Nm[j, i]
                val = zn[ibj]
                for kk in range(3):
                    if Mn[ibj, kk] == j:
                        val -= tov[kk, ibj]
                toc[i, j] = val

        tanhtoc = np.tanh(-toc / 2.0)

        for i in range(N):
            for kk in range(3):
                ichk = Mn[i, kk]
                prod = 1.0
                for ii in range(nrw[ichk]):
                    if Nm[ichk, ii] != i:
                        prod *= tanhtoc[ii, ichk]
                tov[kk, i] = 2.0 * _platanh(-prod)

    return None, -1


# ── Message unpacking (JT77) ──────────────────────────────────────────────────

_C1 = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'   # 37 chars (c1 in unpack28)
_C2 = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'    # 36 chars
_C3 = '0123456789'                               # 10 chars
_C4 = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'             # 27 chars

_NTOKENS = 2063592
_MAX22   = 4194304
_MAXGRID4 = 32400


def _unpack28(n28: int) -> Optional[str]:
    """Decode a 28-bit callsign token to a string (unpack28.f90)."""
    if n28 < _NTOKENS:
        if n28 == 0: return 'DE'
        if n28 == 1: return 'QRZ'
        if n28 == 2: return 'CQ'
        if n28 <= 1002: return f'CQ_{n28-3:03d}'
        if n28 <= 532443:
            n = n28 - 1003
            chars = []
            for _ in range(4):
                chars.append(_C4[n % 27])
                n //= 27
            return 'CQ_' + ''.join(reversed(chars)).strip()
        return None   # reserved token range

    n28 -= _NTOKENS
    if n28 < _MAX22:
        return f'<hash22:{n28}>'   # hash table lookup not available; rare in practice

    # Standard callsign: 6 characters encoded in mixed base
    n = n28 - _MAX22
    i1 = n // (36 * 10 * 27 * 27 * 27)
    n -= i1 * (36 * 10 * 27 * 27 * 27)
    i2 = n // (10 * 27 * 27 * 27)
    n -= i2 * (10 * 27 * 27 * 27)
    i3 = n // (27 * 27 * 27)
    n -= i3 * (27 * 27 * 27)
    i4 = n // (27 * 27)
    n -= i4 * (27 * 27)
    i5 = n // 27
    i6 = n - 27 * i5
    try:
        call = (_C1[i1] + _C2[i2] + _C3[i3] + _C4[i4] + _C4[i5] + _C4[i6]).strip()
    except IndexError:
        return None
    return call if call else None


def _to_grid4(n: int) -> Optional[str]:
    """Decode a 15-bit maidenhead grid locator (to_grid4.f90)."""
    j1 = n // 1800
    n -= j1 * 1800
    j2 = n // 100
    n -= j2 * 100
    j3 = n // 10
    j4 = n - j3 * 10
    if not (0 <= j1 <= 17 and 0 <= j2 <= 17 and 0 <= j3 <= 9 and 0 <= j4 <= 9):
        return None
    return chr(j1 + ord('A')) + chr(j2 + ord('A')) + chr(j3 + ord('0')) + chr(j4 + ord('0'))


def _bits_to_int(bits: np.ndarray, start: int, length: int) -> int:
    """Extract integer from a slice of a bit array (MSB first)."""
    val = 0
    for b in bits[start:start + length]:
        val = (val << 1) | int(b)
    return val


def _unpack77(bits77: np.ndarray) -> Optional[str]:
    """Decode 77 message bits to a human-readable string.

    Handles the message types used in MSK144 contacts:
      i3=1 / i3=2 : standard two-callsign messages (CQ, grid, SNR, RRR, 73)
      i3=0, n3=0  : free text (13 chars)

    Other types return a placeholder rather than None so success is recorded.
    """
    if len(bits77) < 77:
        return None

    # i3 = bits 74..76, n3 = bits 71..73  (0-indexed)
    i3 = _bits_to_int(bits77, 74, 3)
    n3 = _bits_to_int(bits77, 71, 3)

    # Validity gate (mirrors msk144decodeframe.f90)
    invalid = (
        (i3 == 0 and n3 in (1, 3, 4) or n3 > 5) or
        i3 == 3 or i3 > 5
    )
    if invalid:
        return None

    if i3 == 1 or i3 == 2:
        # Standard message: n28a(28) ipa(1) n28b(28) ipb(1) ir(1) igrid4(15) i3(3)
        n28a  = _bits_to_int(bits77,  0, 28)
        ipa   = _bits_to_int(bits77, 28,  1)
        n28b  = _bits_to_int(bits77, 29, 28)
        ipb   = _bits_to_int(bits77, 57,  1)
        ir    = _bits_to_int(bits77, 58,  1)
        igrid = _bits_to_int(bits77, 59, 15)

        call1 = _unpack28(n28a)
        call2 = _unpack28(n28b)
        if call1 is None or call2 is None:
            return None

        suffix = '/R' if i3 == 1 else '/P'
        if ipa and ' ' in call1:
            call1 = call1.rstrip() + suffix
        if ipb and ' ' in call2:
            call2 = call2.rstrip() + suffix

        if call1 == 'CQ_':
            call1 = 'CQ'

        if igrid <= _MAXGRID4:
            grid = _to_grid4(igrid)
            if grid is None:
                return None
            msg = f'{call1} {call2} {grid}'
            if ir:
                msg = f'{call1} {call2} R {grid}'
        else:
            irpt = igrid - _MAXGRID4
            if irpt == 1:   msg = f'{call1} {call2}'
            elif irpt == 2: msg = f'{call1} {call2} RRR'
            elif irpt == 3: msg = f'{call1} {call2} RR73'
            elif irpt == 4: msg = f'{call1} {call2} 73'
            else:
                isnr = irpt - 35
                if isnr > 50:
                    isnr -= 101
                sign = '+' if isnr >= 0 else ''
                crpt = f'{sign}{isnr:02d}' if isnr >= 0 else f'{isnr:03d}'
                msg = f'{call1} {call2} {crpt}'
                if ir:
                    msg = f'{call1} {call2} R{crpt}'

        return msg.strip()

    if i3 == 0 and n3 == 0:
        # Free text: 71 bits → 13 characters (base-42 encoding)
        CHARS = ' 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?'
        n = _bits_to_int(bits77, 0, 71)
        text = []
        for _ in range(13):
            text.append(CHARS[n % 42])
            n //= 42
        return ''.join(reversed(text)).strip() or None

    # Other valid types (i3=2 contest, i3=4, i3=5) — return placeholder
    return f'<MSG i3={i3} n3={n3}>'


# ── Sync template (shared with msk144_spd.py) ─────────────────────────────────

def _build_sync_template() -> np.ndarray:
    """42-sample complex sync template cb = cbi + j*cbq (msk144decodeframe.f90)."""
    s8 = 2 * np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=np.float32) - 1
    pp = np.sin(np.arange(12) * np.pi / 12).astype(np.float32)
    cbi = np.zeros(42, dtype=np.float32)
    cbq = np.zeros(42, dtype=np.float32)
    cbq[0:6]  = pp[6:12] * s8[0]
    cbq[6:18] = pp       * s8[2]
    cbq[18:30]= pp       * s8[4]
    cbq[30:42]= pp       * s8[6]
    cbi[0:12] = pp       * s8[1]
    cbi[12:24]= pp       * s8[3]
    cbi[24:36]= pp       * s8[5]
    cbi[36:42]= pp[0:6]  * s8[7]
    return (cbi + 1j * cbq).astype(np.complex64)

_CB = _build_sync_template()
_PP = np.sin(np.arange(12) * np.pi / 12).astype(np.float32)
_S8 = (2 * np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int32) - 1)


# ── Main decode entry point ───────────────────────────────────────────────────

def msk144decodeframe(
    c: np.ndarray,
) -> tuple[Optional[str], int]:
    """Decode one MSK144 frame.

    Parameters
    ----------
    c : complex64 array, shape (NSPM,) = (864,)
        Complex baseband frame, timing-aligned (symbol boundary at index 0).

    Returns
    -------
    message : str or None  — decoded message, or None on failure
    nharderror : int       — hard-error count (≥0 on success, −1 on failure)
    """
    if len(c) < NSPM:
        return None, -1

    c = c.astype(np.complex64)

    # ── Phase correction (msk144decodeframe.f90 lines 52-59) ──────────────────
    cca = (c[0:42] * np.conj(_CB)).sum()
    ccb = (c[336:378] * np.conj(_CB)).sum()
    phase0 = np.angle(cca + ccb)
    cfac = np.complex64(np.cos(phase0) + 1j * np.sin(phase0))
    c = c * np.conj(cfac)

    # ── Matched filter → 144 soft bits ────────────────────────────────────────
    pp = _PP
    softbits = np.empty(144, dtype=np.float32)

    # Bit 0: imag wrapped around frame boundary
    softbits[0] = ((c[0:6].imag * pp[6:12]).sum() +
                   (c[858:864].imag * pp[0:6]).sum())
    # Bit 1: real, first 12 samples
    softbits[1] = (c[0:12].real * pp).sum()

    # Bits 2..143 (Fortran i=2..72)
    # The 71 imaginary windows are non-overlapping, stride=12, starting at offset 6:
    #   c.imag[6:18], c.imag[18:30], ..., c.imag[846:858]  → reshape(71, 12)
    # The 71 real windows are non-overlapping, stride=12, starting at offset 12:
    #   c.real[12:24], c.real[24:36], ..., c.real[852:864]  → reshape(71, 12)
    # Two matmul calls replace 142 individual np.sum() calls.
    softbits[2::2] = c.imag[6:858].reshape(71, 12) @ pp   # even indices 2,4,...,142
    softbits[3::2] = c.real[12:864].reshape(71, 12) @ pp  # odd  indices 3,5,...,143

    # ── Sync hard-error gate ──────────────────────────────────────────────────
    hardbits = (softbits >= 0).astype(np.int32)
    nbadsync1 = (8 - int(((2*hardbits[0:8]  - 1) * _S8).sum())) // 2
    nbadsync2 = (8 - int(((2*hardbits[56:64] - 1) * _S8).sum())) // 2
    if nbadsync1 + nbadsync2 > 4:
        return None, -1

    # ── Normalise soft bits ────────────────────────────────────────────────────
    sav  = softbits.mean()
    s2av = (softbits ** 2).mean()
    ssig = np.sqrt(max(s2av - sav**2, 1e-10))
    softbits = softbits / ssig

    # ── Extract LLRs (skip sync symbols, scale by channel SNR estimate) ───────
    sigma = 0.60
    llr = np.empty(128, dtype=np.float32)
    llr[0:48]  = softbits[8:56]    # data symbols after first sync block
    llr[48:128]= softbits[64:144]  # data symbols after second sync block
    llr *= 2.0 / (sigma ** 2)

    # ── Belief-propagation decode ─────────────────────────────────────────────
    message77_bits, nharderror = _bpdecode128_90(llr)
    if message77_bits is None:
        return None, -1

    # ── Unpack message ────────────────────────────────────────────────────────
    msg = _unpack77(message77_bits)
    return msg, nharderror
