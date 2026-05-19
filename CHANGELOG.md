# Changelog

All notable changes to MAP144 are recorded here.  Format roughly follows
[Keep a Changelog](https://keepachangelog.com/) — newest first, ham-readable.

Versions are bumped at noticeable-to-tester intervals, not per commit.
Each released version has a matching git tag (e.g. ``v0.1.1-alpha``).

## v0.1.2-alpha — 2026-05-19

### Fixed

- **DAX-IQ stream silently stolen by other Flex clients.**  When WSJT-X
  or a second MAP144 instance created its own DAX-IQ stream on the same
  Flex radio, this MAP144's filter-watch code rewrote its filter to
  follow the foreign stream — and silently stopped receiving IQ (the
  GUI showed "initializing..." while the Flex stream-flow indicators
  for our client went to zero).  On shutdown the cleanup ``stream
  remove`` then targeted the foreign id and was rejected by the radio,
  leaving our own stream orphaned on the radio.

  Fix: ``flexclient/setup.py`` now remembers the stream-id it created
  (``self._own_stream_id``).  Stream-status broadcasts for a different
  stream-id are checked against our slice-id: same slice = radio-
  initiated reassignment (rare; update filter); different slice = a
  foreign client's stream (do NOT update filter; our state untouched).

  Operationally: MAP144 and WSJT-X (and a second MAP144) can now run
  side by side against the same Flex radio, each on its own slice.

## v0.1.1-alpha — 2026-05-19

### Fixed

- **DF column in decode list always showed 0 Hz.**  The block pipeline
  detector wasn't including the within-channel sub-bin frequency offset
  (``fc_offset``) when reporting ``fc_hz`` to the decoder.  As a side
  effect, decoded WAVs were being pre-mixed to channel-centre exactly,
  forcing jt9 to absorb any real signal offset via its FTOL search at
  the cost of decode probability.  Now matches the legacy detector
  formula — sub-Hz pre-correction via the sync-correlator's fine
  frequency-offset estimate is also wired in.

### Added

- **Version is now displayed**:
  - Main window title bar: ``map144 v0.1.1-alpha — 50.260 MHz``
  - Startup log line in ``MSK144/logs/map144_*.log``
  - Install-script "ready" banner

- **CHANGELOG.md** (this file) — versioned record of tester-visible changes.

## v0.1.0-alpha — 2026-05-18

Initial alpha release.  Distributed to the operator's private VHF group.

### Features

- Real-time MSK144 detect + decode pipeline (legacy DSP path is the default).
- 48-channel polyphase channelizer covering ±20 kHz around the MSK144 calling frequency.
- Two-detector design (squared-FFT tone pair + coherent sync-word correlator).
- Decodes reported to PSKReporter, WSJT-X UDP, GridTracker, DXcluster.
- Dual-polarisation support on dual-channel digital receivers (~3 dB advantage on 50 MHz Faraday-rotated pings).
- ``compare_decoders.py`` side-by-side comparison vs WSJT-X.
- Captures browser with substring filter + scrollable list.
- Cross-platform install scripts (``install.sh`` / ``install.ps1``); cross-platform launchers (``run.sh`` / ``run.bat``).
- Block-pipeline architecture available behind ``MAP144_USE_BLOCKS=1`` (opt-in only; not recommended for general use yet).

### Tested with

FlexRadio 6000 series (DAX-IQ), USRP B210, Airspy HF+, NESDR Smart.
