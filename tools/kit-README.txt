MAP144 B210 install kit
=======================

Self-contained installer for MAP144 with USRP B210 support on Windows 11.
No internet required at install time; everything needed is in this directory.

WHAT'S IN THIS KIT
------------------
  install-b210.bat       <- double-click this to install
  install-b210.ps1       <- the actual installer (called by the .bat)
  README-INSTALL.txt     <- this file
  VERSION.txt            <- version info (map144 git SHA, build date)
  ettus/                 <- Ettus UHD Windows installer
  map144-env.zip         <- portable Python env with UHD + map144 deps
  map144-src.zip         <- MAP144 source code snapshot

PREREQUISITES (on the target PC)
--------------------------------
  - Windows 11 (or Windows 10 build 17763+)
  - Administrator access (UAC prompt during install)
  - 3 GB free disk space on C:
  - WSJT-X installed separately (provides the jt9 decoder) — install from
    https://wsjt.sourceforge.io if not already present.

The kit does NOT need:
  - Internet access (everything is in this directory)
  - Pre-installed Python, Miniconda, or UHD
  - Visual Studio or any developer tooling

HOW TO INSTALL
--------------
  1. Plug the B210 into a USB 3.0 port (blue USB port).
  2. Copy this entire kit directory to the target PC (or leave on USB stick).
  3. Double-click install-b210.bat.
  4. Approve the UAC prompt ("Yes" to administrator privileges).
  5. Watch the progress output.  Total time: ~5 minutes.
  6. The last line will say either:
        READY         <- success; launch with C:\map144\run-b210.bat
        FAILED: ...   <- see the FAILED line for the reason

DEFAULT INSTALL LOCATION
------------------------
  C:\map144\            <- MAP144 source + run-b210.bat
  C:\map144\env\        <- portable Python env (~1 GB after extract)
  C:\map144\MSK144\     <- decode WAVs + JSONL logs (created on first run)
  C:\Program Files\UHD\ <- Ettus UHD (driver + firmware images)

To install elsewhere, run from PowerShell:
  powershell -ExecutionPolicy Bypass -File install-b210.ps1 -InstallPath D:\map144

HOW TO LAUNCH
-------------
  Double-click C:\map144\run-b210.bat
  (or whichever drive you installed to)

  MAP144 will start with Flex as the default source.  Switch to USRP from
  the source menu to stream from the B210.

RE-INSTALL OVER AN EXISTING INSTALL
-----------------------------------
  Re-running install-b210.bat with the same kit version is a no-op.
  Re-running with a different kit version prompts to overwrite.
  Pass -Force to skip the prompt (e.g. for unattended re-installs).

TROUBLESHOOTING
---------------
  Install log:    C:\map144\install.log
  MAP144 logs:    C:\map144\MSK144\logs\map144_*.log

  "FAILED: Ettus installer exit code N" — try running ettus\*.exe by hand
    (right-click, Run as administrator) to see the actual error dialog.

  "B210 not detected" warning — the B210 wasn't plugged in (or wasn't on
    a USB 3 port) when the smoke test ran.  Plug it in and launch
    run-b210.bat; it will re-probe.

  Anything else — see docs/ALPHA_NOTES.md inside the map144 source tree
  (C:\map144\docs\ALPHA_NOTES.md after install) for the manual recipe.
