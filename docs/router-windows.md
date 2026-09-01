# B210 Audio / IQ Router — Windows install & start

Step-by-step for a Windows PC with a USRP B210. Goal: select WSJT-X dial
frequencies (FT8, FT4, MSK144, …), stream each to a virtual soundcard, and
point one or more WSJT-X instances at those inputs. Optional: MAP65 / QMAP
wideband IQ over UDP.

The router **owns the B210** while it is running. Do not also run MAP144
against the same radio at the same time.

---

## What you need

| Item | Notes |
|---|---|
| Windows 10 (17763+) or Windows 11 | Prefer a non-OneDrive path such as `C:\map144` |
| USRP B210 | USB **3.0** port (blue) |
| MAP144 with B210 support | Install kit **or** manual UHD + conda (below) |
| WSJT-X | Separate install; for decoding the audio feeds |
| Virtual audio cable | **VB-CABLE** (free) for one dial; **VB-CABLE A+B** or VoiceMeeter for several |
| Administrator access | Needed once for UHD / kit install |

---

## 1. Install MAP144 + B210 support

### Option A — `install.ps1` (recommended for router on a networked PC)

From a git checkout of `main`:

```powershell
cd C:\map144
.\install.ps1
```

This will:

1. Create `.venv` and install Python deps (`numpy==1.26.4`, …)
2. Download/install **Ettus UHD** (admin UAC) including **B210 firmware**
   images under `C:\Program Files\UHD\share\uhd\images\`  
   (skip with `.\install.ps1 -SkipUhd` if you only want Flex/audio work)

**Python `import uhd`:** Windows has no pip wheel. After `install.ps1`, also do
**one** of:

- Offline kit `env\` (Option B), or  
- Miniconda: `conda create -n map144 python=3.11` then  
  `conda install -c conda-forge uhd` and `pip install -r requirements.txt`

`run-router.bat` auto-detects `env\`, `.venv\`, or the conda `map144` env.

Standalone UHD/firmware only:

```powershell
powershell -ExecutionPolicy Bypass -File tools\Install-UhdWindows.ps1
```

### Option B — B210 install kit (offline / contest PC)

Same kit used for MAP144. Details: `docs/ALPHA_NOTES.md` and
`tools/kit-README.txt`.

1. Plug the B210 into a USB 3.0 port.
2. Copy the kit folder to the PC (or leave it on a USB stick).
3. Double-click `install-b210.bat` → approve UAC → wait until it prints `READY`.
4. Default layout after install:
   - `C:\map144\` — source + launchers  
   - `C:\map144\env\` — Python env with UHD bindings  
   - `C:\Program Files\UHD\` — Ettus driver / firmware  

### Option C — already have MAP144 + B210 working

If `run-b210.bat` already works, skip to §2.

---

## 2. Install a virtual audio cable

Windows cannot create PipeWire-style null sinks from user space. Install a
driver that presents a fake soundcard:

| Need | Install |
|---|---|
| One WSJT-X dial | [VB-CABLE](https://vb-audio.com/Cable/) (free) |
| Two+ dials (or RF0+RF1) | [VB-CABLE A+B](https://vb-audio.com/Cable/) or VoiceMeeter |

After install, reboot if the installer asks. In Windows Sound settings you
should see devices such as:

- `CABLE Input` / `CABLE Output` (single cable), or  
- `Cable A Input` / `Cable A Output`, `Cable B Input` / `Cable B Output`

**MAP144/router writes to the Input side. WSJT-X listens on the Output side.**

---

## 3. Install `sounddevice` in the MAP144 env

PortAudio bindings are required on Windows (listed in `requirements.txt`).

**Kit / portable env:**

```bat
C:\map144\env\python.exe -m pip install sounddevice
```

**Conda env:**

```bat
conda activate map144
pip install sounddevice
```

Quick check:

```bat
C:\map144\env\python.exe -c "import sounddevice as sd; print([d['name'] for d in sd.query_devices() if d['max_output_channels']>0])"
```

You should see `CABLE Input` or `Cable A Input` in that list.

---

## 4. Start the router

```bat
cd C:\map144
run-router.bat
```

`run-router.bat` finds Python in this order: `env\`, `.venv\`, then
`%USERPROFILE%\miniconda3\envs\map144\`.

The GUI opens: **MAP144 — B210 Audio / IQ Router**.

---

## 5. Configure and Apply

1. **Band** — e.g. `2m` or `6m`.
2. Check the **dial channels** you want (or **Select common** for MSK144+FT8+FT4).
3. On the right: set **RF0 gain** (and RF1 if dual) and **Noise Blanker**
   backend / K — same idea as MAP144’s B210 and IQ/NB panels. These stay
   live after Apply.
4. Optional: enable **MAP65** and/or **QMAP** (host / port / centre). Defaults
   are `127.0.0.1:50002` (MAP65) and `:50004` (QMAP).
5. Optional: **Dual RF ports** if both B210 RX ports are in use.
6. Read the **Plan** line (pan MHz, sample rate, span). If it shows an error,
   deselect channels that do not fit in one 192 kHz IF.
7. Click **Apply**.

Status text should show the router running. Leave this window open while
operating. Gain and blanker can be adjusted without stopping.

---

## 6. Point WSJT-X at the cable

For each WSJT-X instance:

1. **File → Settings → Audio → Input**  
   - Single VB-CABLE → **`CABLE Output`**  
   - Cable A/B → **`Cable A Output`** / **`Cable B Output`**  
2. **Radio** → None (or whatever you use with no CAT).  
3. Set **mode** and **dial frequency** to match the channel you selected in
   the router (e.g. MSK144 @ 144.150, FT8 @ 144.174).  
4. Confirm the WSJT-X level meter is alive (~30 dB noise is typical).

### Multiple dials

Each selected dial needs its own virtual cable (and its own WSJT-X instance).
Assign cables explicitly if auto-match is wrong:

```bat
set MAP144_WSJTX_DEVICE=Cable A Input, Cable B Input
run-router.bat
```

Or per RF port:

```bat
set MAP144_WSJTX_DEVICE_RF0=Cable A Input
set MAP144_WSJTX_DEVICE_RF1=Cable B Input
```

Then in WSJT-X #1 use Cable A **Output**, in WSJT-X #2 use Cable B **Output**.

---

## 7. Stop

In the router window click **Stop**, or close the window. That releases the
B210 and stops writing to the cables.

---

## Checklist (first successful run)

- [ ] `uhd_find_devices` sees the B210 (or kit install said READY)
- [ ] VB-CABLE (or A+B) installed; Output device visible in Windows Sound
- [ ] `import sounddevice` works in the map144 env
- [ ] `run-router.bat` opens the GUI
- [ ] Apply with one dial selected succeeds (no PortAudio / device error)
- [ ] WSJT-X Input = **CABLE Output** (or Cable A Output); meter moves
- [ ] A known-good signal or strong local decode appears in WSJT-X

---

## Troubleshooting

| Symptom | What to try |
|---|---|
| `run-router.bat` → no Python env | Install the B210 kit, or point at conda `map144` |
| Apply fails: device not found | Install VB-CABLE; run the `sounddevice` device list in §3; set `MAP144_WSJTX_DEVICE` |
| Apply fails: `sounddevice` missing | `env\python.exe -m pip install sounddevice` |
| WSJT-X meter dead | Wrong side of the cable (use **Output**, not Input); wrong WSJT-X instance; router not Applied |
| B210 not found | USB 3 port; Device Manager → USRPs; power-cycle B210 |
| Plan ERROR: span too wide | Fewer dials, or turn off MAP65/QMAP if the window does not fit 192 kHz |
| Glitches / warble over time | Clock drift vs virtual cable — known limitation vs Linux PipeWire; restart Apply if needed |

Logs: if MAP144 logging dirs exist, see `C:\map144\MSK144\logs\`. Router
messages also print in the console window behind `run-router.bat`.

---

## Related docs

- Overview / Linux / architecture: [`router.md`](router.md)
- B210 kit & UHD: [`ALPHA_NOTES.md`](ALPHA_NOTES.md)
- Kit quick card: `tools/kit-README.txt`
