# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Single-letter shortcuts for fast annotation in tools/review_corpus.py.

Edit freely.  The review UI displays this mapping on screen and listens
for the corresponding keypresses.  Case-sensitive: 'M' is distinct from
'm'.  Keep keys to characters easy to remember by description.

A tag named here does NOT need to exist in :mod:`tests.corpus.tags`.  If
absent there, the annotation creates an "undeclared" tag — visible in
the gallery and surfaced by ``tools/audit_taxonomy.py`` as a promotion
candidate for the vocabulary.

The starter set below maps to the description axes laid out for the
gallery review workflow.  Expect to refine after live use.
"""

#
# Design principle: prefer BEHAVIOR tags (observable in the spectrogram /
# metric panel) over INFERRED propagation modes.  Operator review can't
# reliably tell tropo from aircraft-scatter from terrestrial-multipath from
# one 12-kHz spectrogram — that's classifier work, not annotation work.
#
# Mode-naming tags that *are* in this list are the ones distinguishable on
# behaviour:
#
#   meteor_ping   — discrete impulsive burst with classic ramp-down tail
#   weak_fading   — slow time-varying weak signal of unknown specific mode
#                   (covers tropo / aircraft-enhanced / terrestrial-multipath)
#   groundwave    — short-distance signal that is weak because of antenna
#                   geometry or low power, not propagation
#
# If you do know the specific mode (band open to Es, known aircraft pass,
# etc.), add the more-specific tag through the "new tag" input — those
# annotations are still valid, just not on the rapid-fire keyboard row.
#
SHORTCUTS: dict[str, str] = {
    # ── Temporal axis (observable) ───────────────────────────────────────
    'b': 'brief_burst',         # very short MS-like ping (<= 500 ms)
    's': 'short_burst',         # MS-like (<= 1 s)
    'm': 'multi_frame',         # multiple MSK144 frames decoded
    'c': 'continuous',          # signal present through most of period

    # ── Spectral axis (observable) ───────────────────────────────────────
    'f': 'fluttery',            # σ_f >= 1.7 Hz, multipath-like flutter
    'd': 'drifting',            # linear f drift across burst
    'x': 'multipath',           # A and f correlated (two-ray signature)

    # ── Mode tags distinguishable on behaviour ───────────────────────────
    'M': 'meteor_ping',         # classic random MS ping (uppercase M)
    'w': 'weak_fading',         # catch-all weak slow-varying (tropo / AS /
                                # terrestrial multipath / unknown)
    'g': 'groundwave',          # short distance, weak from geometry not
                                # propagation

    # ── Position in 15 s period (observable) ─────────────────────────────
    'e': 'edge_early',          # dt < 1 s
    'l': 'edge_late',           # dt > 13 s
    'k': 'consistent_dt_skew',  # dt repeated across same-callsign decodes
                                # → tx clock offset suspect

    # ── Quality flags (observable / inferred-but-distinct) ───────────────
    'p': 'phantom_suspect',     # decoded callsign / grid looks malformed
    'q': 'clock_skew_suspect',  # see consistent_dt_skew
    'r': 'receiver_issue',      # AGC pump, spur, splatter from local
    'C': 'co_channel',          # interfering signal in same channel
    'S': 'splatter',            # bleed from adjacent channel

    # ── Comparison / canonical cases ─────────────────────────────────────
    'P': 'matches_105430',      # canonical "barely visible" weak ping
}
