"""Which application currently has focus, so dictation can be formatted to suit it.

Pure ctypes against user32 and kernel32 - no pywin32, no psutil. Every call fails soft:
an elevated foreground process cannot be opened from a normal-privilege process, and that
is a normal outcome rather than an error.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

from loguru import logger

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
MAX_PATH_LONG = 32768

user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

# Formatting contexts. The exe name decides; the window title is only logged.
CODE = "code"
CHAT = "chat"
EMAIL = "email"
TERMINAL = "terminal"
PROSE = "prose"

_EXE_CONTEXT = {
    "code.exe": CODE, "devenv.exe": CODE, "idea64.exe": CODE, "pycharm64.exe": CODE,
    "sublime_text.exe": CODE, "cursor.exe": CODE, "windsurf.exe": CODE,
    "rider64.exe": CODE, "clion64.exe": CODE, "webstorm64.exe": CODE,
    "slack.exe": CHAT, "discord.exe": CHAT, "teams.exe": CHAT, "ms-teams.exe": CHAT,
    "telegram.exe": CHAT, "whatsapp.exe": CHAT, "signal.exe": CHAT,
    "outlook.exe": EMAIL, "thunderbird.exe": EMAIL, "olk.exe": EMAIL,
    "windowsterminal.exe": TERMINAL, "wt.exe": TERMINAL, "cmd.exe": TERMINAL,
    "powershell.exe": TERMINAL, "pwsh.exe": TERMINAL, "alacritty.exe": TERMINAL,
    "conhost.exe": TERMINAL, "mintty.exe": TERMINAL,
    "winword.exe": PROSE, "notepad.exe": PROSE, "obsidian.exe": PROSE,
    "notion.exe": PROSE, "msedge.exe": PROSE, "chrome.exe": PROSE,
    "firefox.exe": PROSE, "brave.exe": PROSE,
}


@dataclass
class Foreground:
    exe: Optional[str]      # 'Code.exe', or None if it could not be determined
    title: Optional[str]
    context: str            # one of the constants above
    elevated_guess: bool    # True when the process could not be opened at all

    @property
    def usable(self) -> bool:
        """False means synthetic keystrokes will probably be blocked by UIPI."""
        return not self.elevated_guess


def _window_title(hwnd) -> Optional[str]:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return None
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value or None


def _process_exe(hwnd) -> tuple[Optional[str], bool]:
    """Returns (exe basename, could_not_open). could_not_open usually means elevated."""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return None, False

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        # Access denied: the foreground process runs at a higher integrity level.
        return None, True
    try:
        size = wintypes.DWORD(MAX_PATH_LONG)
        buf = ctypes.create_unicode_buffer(MAX_PATH_LONG)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None, False
        return buf.value.rsplit("\\", 1)[-1], False
    finally:
        kernel32.CloseHandle(handle)


def foreground() -> Foreground:
    """Inspect the focused window. Never raises."""
    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return Foreground(None, None, PROSE, False)
        title = _window_title(hwnd)
        exe, could_not_open = _process_exe(hwnd)
        context = _EXE_CONTEXT.get((exe or "").lower(), PROSE)
        if could_not_open:
            logger.warning(
                "Foreground process could not be opened; it is probably elevated. "
                "Synthetic paste will be blocked by UIPI."
            )
        return Foreground(exe, title, context, could_not_open)
    except Exception as exc:
        logger.exception(f"Foreground detection failed: {exc}")
        return Foreground(None, None, PROSE, False)
