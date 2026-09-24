"""Launch at login, via the registry Run key.

winreg is in the standard library, so this needs no pywin32 and no .lnk shortcut. The
command uses pythonw.exe rather than python.exe, otherwise every login opens a console
window behind the tray icon.
"""

from __future__ import annotations

import sys
import winreg
from pathlib import Path

from loguru import logger

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "Transcriber"


def _command() -> str:
    """The command Windows should run at login, quoted for the registry."""
    # sys.executable is .venv\Scripts\python.exe when run normally; the windowed twin
    # sits beside it. Fall back to whatever launched us if it is missing.
    exe = Path(sys.executable)
    windowed = exe.with_name("pythonw.exe")
    interpreter = windowed if windowed.exists() else exe
    return f'"{interpreter}" -m app.main'


def is_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _type = winreg.QueryValueEx(key, VALUE_NAME)
            return bool(value)
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning(f"Could not read the autostart entry: {exc}")
        return False


def set_enabled(enabled: bool) -> bool:
    """Add or remove the Run entry. Returns whether the change took effect."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            if enabled:
                command = _command()
                winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)
                logger.info(f"Autostart enabled: {command}")
            else:
                try:
                    winreg.DeleteValue(key, VALUE_NAME)
                    logger.info("Autostart disabled.")
                except FileNotFoundError:
                    pass  # already absent, which is the requested state
        return True
    except OSError as exc:
        logger.warning(f"Could not change the autostart entry: {exc}")
        return False
