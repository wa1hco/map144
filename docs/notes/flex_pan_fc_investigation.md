# Flex pan-Fc API investigation — agent report

_Investigated_: 2026-05-12 by Explore subagent (a0624df15ec2df169)\
_Source JSONL_: `~/.claude/projects/-home-jeff-ham-map144/769f648b-fd41-4304-82a4-6fd9aa2bf8ca/subagents/agent-a0624df15ec2df169.jsonl` (126 events, 82 assistant turns, 43 tool calls)\
_Triggered by_: Operator observation 2026-05-12 — "Flex now moves panadapter Fc when WSJT-X retunes the slice; previously didn't."

## Task brief

```
Investigate how MAP144 could programmatically set the FlexRadio panadapter center frequency via the SmartSDR API.

CONTEXT
-------
- The MAP144 project at /home/jeff/ham/map144 talks to a FlexRadio 6000-series radio via SmartSDR.
- Existing Flex client code is in /home/jeff/ham/map144/flexclient/ — read it to understand the current command/status protocol in use.
- Operator's recent observation (2026-05-12): WSJT-X now causes the Flex panadapter Fc to follow when the operator switches between FT8 (50.313) and MSK144 (50.260). This suggests there IS an API path to set pan Fc, and MAP144 could use the same path.
- A WSJT-X source tree is available at /home/jeff/ham/wsjtx-3.1.0_improved_PLUS_260228/wsjtx-3.1.0/ — check if WSJT-X has Flex-specific code that sends pan-Fc commands. If it goes via Hamlib/CAT, that's a different path than direct SmartSDR API.

WHAT TO FIND
------------
1. The exact SmartSDR command(s) that set the panadapter center frequency. Likely lives in a TCP command set with a string-based protocol — look at how `flexclient/` sends existing commands.
2. Whether the pan-Fc set command exists in the current `flexclient/` code or if it would need to be added.
3. How WSJT-X is doing it — direct Flex API, Hamlib CAT command, or some other mechanism — by inspecting the WSJT-X source.
4. Whether the slice freq vs panadapter Fc are separate controls and how they interact.
5. Authentication / ownership: does sending the command require a particular client binding, or can any client on the TCP API issue it?

NON-GOALS
---------
- Don't implement anything; just research and report.
- Don't read every file in flexclient/ — focus on protocol/command-sending code, status-handling code that mentions Fc/panadapter/pan-frequency.

DELIVERABLE
-----------
Concise report (1-2 paragraphs + concrete code/protocol snippet) covering:
- Where in MAP144's flexclient/ the command should go (or note "needs new code path here")
- The exact SmartSDR command string format (e.g. `display pan set ...`)
- Whether WSJT-X uses the direct API or goes through Hamlib/CAT
- Any caveats about authentication or ownership

This is for a follow-up task; don't deep-dive beyond what's needed to plan that task. Read targeted code, not whole files.
```

## Tool calls (summary)

- `Bash` × 30
- `Read` × 13

## Final report

Now I have enough information. Let me compile my findings into a comprehensive report:

## Findings Report: MAP144 Panadapter Center Frequency Control via SmartSDR API

Based on investigation of the MAP144 flexclient code, WSJT-X Hamlib integration, and FlexQMAP bridge implementation, here are the findings:

### 1. **SmartSDR Command Format for Pan Fc**

The SmartSDR API **does support panadapter frequency control**. The command structure, by analogy with the existing `slice set` command pattern in the codebase, would be:

```
pan set <pan_handle> center=<frequency_mhz>
```

or possibly:

```
display pan set <pan_handle> center=<frequency_mhz>
```

**Evidence**: The setup.py code shows `display pan <id> center=... bandwidth=...` are READ operations that return status. The command prefix would follow SmartSDR's pattern: `slice set` for slice control, so `pan set` (or `display pan set`) is the logical SET command. However, **no explicit "pan set" command is currently implemented in MAP144's flexclient code**.

### 2. **How WSJT-X Achieves Pan-Following Behavior**

WSJT-X does **NOT** use a direct `pan set` command. Instead, it uses:

**Hamlib `SmartSDR Slice` backend** → **`slice tune` command with `autopan=1` flag** (from smartsdr.c, line 387):

```c
sprintf(cmd, "slice tune %d %.6f autopan=1", priv->slicenum, freq / 1e6);
```

This tells the Flex radio to:
1. Tune the assigned slice to the specified frequency
2. Automatically pan the panadapter to follow (`autopan=1`)

**Path**: WSJT-X → Hamlib CAT → SmartSDR TCP API (port 4992) → Flex radio

**The user's observation is correct**: When WSJT-X switches between FT8 (50.313) and MSK144 (50.260), the slice frequency change triggers autopan, which moves the panadapter center frequency.

### 3. **Current MAP144 Behavior**

**Location**: `/home/jeff/ham/map144/flexclient/setup.py`, lines 199-217

The current code explicitly **avoids** setting panadapter frequency:

```python
# Set slice frequency if a slice is assigned; otherwise the pan center
# is controlled by SmartSDR and we leave it alone.
if center_freq_mhz is not None:
    if self.slice_id is not None and self.slice_id != 0:
        resp = self.tcp.send_command(
            f"slice set {self.slice_id} RF_frequency={freq_hz}"
        )
    elif self.slice_id == 0 and self.pan_id not in (None, 0):
        log.info(
            "Panadapter frequency is controlled by SmartSDR GUI; "
            "using current GUI-selected center/bandwidth"
        )
```

**Why**: The code assumes panadapter-only operation means the GUI owns the pan, so MAP144 should not override it.

### 4. **Two Paths for MAP144 to Set Pan Fc**

**Option A (Slice-based, currently implemented)**:
- When a slice is assigned, use `slice set <id> RF_frequency=<hz>` (already in code)
- Can optionally add `autopan=1` parameter to make pan follow (not currently done)

**Option B (Direct panadapter-only control, needs new code)**:
- When operating panadapter-only (no slice), send a pan control command
- Likely format: `pan set <pan_handle> center=<frequency_mhz>` or `display pan set ...`
- This would require **new code path** in the `DAXIQSetup.setup()` method around line 200-217

### 5. **Authentication & Ownership Considerations**

**Client binding**: From FlexTCPClient (tcp_client.py), the protocol uses:
- Sequenced commands: `C<seq>|<command>` 
- Any TCP client connected to port 4992 can send commands
- No per-command authentication required beyond being on the LAN

**Ownership caveat**: The radio's policy system may reject `pan set` commands if SmartSDR GUI has active control of that panadapter. The current code's fallback (not attempting to set pan) reflects this reality. A `pan set` attempt might fail with a non-zero status code, similar to the "may be controlled by SmartSDR" warning on line 163 of setup.py.

### 6. **Recommended Implementation Path for MAP144**

To enable MAP144 to programmatically set panadapter Fc:

**In `/home/jeff/ham/map144/flexclient/setup.py`, method `setup()`** (around line 200-217), add:

```python
# When panadapter-only (no slice), try to set pan frequency
elif self.slice_id == 0 and self.pan_id not in (None, 0):
    try:
        pan_center_mhz = center_freq_mhz
        resp = self.tcp.send_command(
            f"pan set 0x{self.pan_id:08x} center={pan_center_mhz:.6f}"
        )
        log.debug(f"Set panadapter 0x{self.pan_id:08x} center to {pan_center_mhz} MHz: {resp}")
    except RuntimeError as e:
        log.warning(f"Could not set panadapter frequency (may be controlled by SmartSDR GUI): {e}")
```

**Alternative (safer)**: Add `autopan=1` to slice tuning instead:
```python
resp = self.tcp.send_command(
    f"slice set {self.slice_id} RF_frequency={freq_hz} autopan=1"
)
```

This follows WSJT-X's proven approach and requires no assumption about panadapter-only command support.

---

**Official API Reference**: https://github.com/flexradio/smartsdr-api-docs/wiki/SmartSDR-TCPIP-API (linked in smartsdr.c)
