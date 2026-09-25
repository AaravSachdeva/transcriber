"""What a dictation ended up as, read back from the text box it was pasted into.

After a paste, the text box is read through Windows UI Automation once a second, and the
pasted span is followed while the user edits it. When focus leaves the box, the next
dictation starts, or a minute passes, whatever the span holds is the final text. Only
that span is kept; the rest of the box is never stored.

Pure ctypes COM against UIAutomationCore, like winctx.py: no comtypes, no pywin32. Every
call fails soft. An app that exposes nothing readable simply yields no final text.

All UI Automation runs on the watcher's own thread. A UIA call is a cross-process call
into the target app and can block, and the GUI thread must never wait on another app.
"""

from __future__ import annotations

import ctypes
import queue
import threading
import time
import uuid
from ctypes import POINTER, byref, c_int, c_void_p, wintypes
from difflib import SequenceMatcher
from typing import Callable, Optional

from loguru import logger

from .winctx import Foreground, user32

ole32 = ctypes.OleDLL("ole32")
oleaut32 = ctypes.WinDLL("oleaut32")
oleaut32.SysStringLen.argtypes = [c_void_p]
oleaut32.SysStringLen.restype = ctypes.c_uint
oleaut32.SysFreeString.argtypes = [c_void_p]
oleaut32.VariantClear.argtypes = [c_void_p]

POLL_S = 1.0        # how often the text box is read while a dictation is watched
WATCH_S = 60.0      # how long after the paste an edit still counts as a correction
MAX_CHARS = 100_000  # a bigger box is a document, and diffing it every second is not free
MIN_MATCH = 0.8     # how closely the span found must resemble what was pasted
PASTE_POLLS = 3     # polls allowed for the paste to show up in the box

UIA_TEXT_PATTERN = 10014
UIA_IS_VALUE_PATTERN_AVAILABLE = 30043
UIA_VALUE_VALUE = 30045
VT_BOOL, VT_BSTR = 11, 8


class _GUID(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _guid(text: str) -> _GUID:
    return _GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


CLSID_CUIAutomation = _guid("ff48dba4-60ef-4201-aa87-54103eef594e")
IID_IUIAutomation = _guid("30cbe57d-d9d0-452a-ab13-7ac5ac4825ee")
IID_IUIAutomationTextPattern = _guid("32eba289-3583-42c9-9c59-3b6d9a1e9b6a")


class _Variant(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3),
                ("value", c_void_p), ("value2", c_void_p)]


def _method(ptr: c_void_p, slot: int, *argtypes):
    """A COM method by its vtable slot, from UIAutomationClient.h. Call it with the
    interface pointer first. A failed HRESULT raises OSError."""
    vtable = ctypes.cast(ptr, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtable[slot])


def _release(ptr: Optional[c_void_p]) -> None:
    if ptr:
        vtable = ctypes.cast(ptr, POINTER(POINTER(c_void_p))).contents
        ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(vtable[2])(ptr)


def _take_bstr(bstr: c_void_p) -> str:
    if not bstr:
        return ""
    try:
        return ctypes.wstring_at(bstr, oleaut32.SysStringLen(bstr))
    finally:
        oleaut32.SysFreeString(bstr)


def _property(element: c_void_p, prop: int):
    """A string or boolean UIA property; None for any other type."""
    var = _Variant()
    _method(element, 10, c_int, POINTER(_Variant))(element, prop, byref(var))
    try:
        if var.vt == VT_BSTR:
            return ctypes.wstring_at(var.value, oleaut32.SysStringLen(var.value)) \
                if var.value else ""
        if var.vt == VT_BOOL:
            return bool((var.value or 0) & 0xFFFF)
        return None
    finally:
        oleaut32.VariantClear(byref(var))


def _text(element: c_void_p) -> Optional[str]:
    """The text in an element: TextPattern where it has one (rich editors, browsers),
    else ValuePattern (plain edit boxes). None when it has neither, or is too big."""
    pattern, document = c_void_p(), c_void_p()
    try:
        _method(element, 14, c_int, POINTER(_GUID), POINTER(c_void_p))(
            element, UIA_TEXT_PATTERN, byref(IID_IUIAutomationTextPattern), byref(pattern))
        if pattern:
            _method(pattern, 7, POINTER(c_void_p))(pattern, byref(document))
            bstr = c_void_p()
            _method(document, 12, c_int, POINTER(c_void_p))(document, MAX_CHARS + 1, byref(bstr))
            text = _take_bstr(bstr)
        elif _property(element, UIA_IS_VALUE_PATTERN_AVAILABLE) is True:
            # Checked first because an element without ValuePattern still reports its
            # value as "", which would read as an empty text box.
            text = _property(element, UIA_VALUE_VALUE) or ""
        else:
            return None
    except OSError:
        return None
    finally:
        _release(document)
        _release(pattern)
    if len(text) > MAX_CHARS:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _Uia:
    """The IUIAutomation object. Create and use it on one thread only."""

    def __init__(self) -> None:
        try:
            ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except OSError:
            pass  # already initialised on this thread, in another mode; UIA still works
        self._ptr = c_void_p()
        ole32.CoCreateInstance(byref(CLSID_CUIAutomation), None, 1,  # CLSCTX_INPROC_SERVER
                               byref(IID_IUIAutomation), byref(self._ptr))

    def focused(self) -> Optional[c_void_p]:
        element = c_void_p()
        try:
            _method(self._ptr, 8, POINTER(c_void_p))(self._ptr, byref(element))
        except OSError:
            return None
        return element if element else None

    def same(self, a: c_void_p, b: c_void_p) -> bool:
        result = wintypes.BOOL()
        try:
            _method(self._ptr, 3, c_void_p, c_void_p, POINTER(wintypes.BOOL))(
                self._ptr, a, b, byref(result))
        except OSError:
            return False
        return bool(result.value)


# --- following the pasted span, kept free of COM so it can be tested directly ---

def _common(a: str, b: str) -> tuple[int, int]:
    """Lengths of the common prefix and common suffix of a and b, not overlapping."""
    limit = min(len(a), len(b))
    prefix = 0
    while prefix < limit and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1
    return prefix, suffix


def locate(before: str, after: str, inserted: str) -> tuple[int, int]:
    """The span of `after` that pasting `inserted` into `before` produced. Found from the
    box's own text, so it holds when the app changed what was pasted (CRLF, accepted
    mentions)."""
    prefix, suffix = _common(before, after)
    start, end = prefix, len(after) - suffix
    # The bare difference is smaller than the paste when the paste replaced a similar
    # selection ("tomorow" -> "tomorrow" differs by one "r"). Widen it to the pasted
    # text when that is there verbatim and covers it; this window only admits such.
    at = after.find(inserted, max(0, end - len(inserted)), start + len(inserted))
    return (at, at + len(inserted)) if at != -1 else (start, end)


def track(prev: str, cur: str, start: int, end: int) -> Optional[tuple[int, int]]:
    """Where the span prev[start:end] is in cur. None when an edit crossed its edge or
    it is gone: the box was cleared, or the message sent.

    Typing at either edge counts as new text, not as a correction.
    ponytail: one diff per poll, so a fix and new typing inside the same second merge,
    and in a run of identical characters an edge can land one character off. Diff at
    keystroke rate (UIA text-changed events) if the finals come out noisy.
    """
    prefix, suffix = _common(prev, cur)
    changed_end = len(prev) - suffix  # prev[prefix:changed_end] became cur[prefix:-suffix]
    delta = len(cur) - len(prev)
    if changed_end <= start:
        return start + delta, end + delta
    if prefix >= end:
        return start, end
    if start <= prefix and changed_end <= end and end + delta > start:
        return start, end + delta
    return None


def _matches(span: str, inserted: str) -> bool:
    a, b = " ".join(span.split()), " ".join(inserted.split())
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= MIN_MATCH


class FieldWatcher:
    """Follows the text box of the latest dictation, on its own thread.

    The controller calls before() when a recording goes to Whisper, which reads the box
    while transcription runs, and after() once the text is pasted. `on_final` receives
    (history id, final text, "implicit" or "unedited").
    """

    def __init__(self, on_final: Callable[[int, str, str], None]) -> None:
        self._on_final = on_final
        self._queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="field-watcher", daemon=True).start()

    def before(self, fg: Foreground) -> None:
        self._queue.put(("before", fg))

    def after(self, fg: Foreground, row_id: int, inserted: str) -> None:
        """`fg` must be the one given to before(): it ties the paste to its snapshot."""
        self._queue.put(("after", fg, row_id, inserted))

    def stop(self) -> None:
        self._queue.put(("stop",))

    def _run(self) -> None:
        try:
            uia = _Uia()
        except OSError as exc:
            logger.warning(f"UI Automation is unavailable; corrections will not be read back: {exc}")
            return
        pre: Optional[dict] = None    # the box before the paste
        watch: Optional[dict] = None  # the box since the paste
        next_poll = 0.0
        while True:
            try:
                command = self._queue.get(
                    timeout=max(0.0, next_poll - time.monotonic()) if watch else None)
            except queue.Empty:
                try:
                    keep = self._poll(uia, watch)
                except Exception as exc:
                    logger.exception(f"Correction capture failed: {exc}")
                    keep = False
                if not keep:
                    self._finish(watch)
                    watch = None
                next_poll = time.monotonic() + POLL_S
                continue

            if watch:
                self._finish(watch)
                watch = None
            kind = command[0]
            if kind == "stop":
                return
            if kind == "before":
                if pre:
                    _release(pre["element"])
                pre = self._snapshot(uia, command[1])
            elif kind == "after" and pre and pre["fg"] is command[1]:
                _fg, row_id, inserted = command[1:]
                watch = {**pre, "row_id": row_id, "inserted": inserted, "span": None,
                         "polls": 0, "deadline": time.monotonic() + WATCH_S,
                         "outcome": "paste not seen in the box"}
                pre = None
                # Read at once: a chat message sent within the first second would
                # otherwise be gone before the paste was ever seen.
                next_poll = 0.0

    @staticmethod
    def _snapshot(uia: _Uia, fg: Foreground) -> Optional[dict]:
        hwnd = user32.GetForegroundWindow()
        element = uia.focused()
        text = _text(element) if element else None
        if text is None:
            _release(element)
            logger.info(f"Correction capture: nothing readable in {fg.exe or 'an unknown app'}.")
            return None
        return {"fg": fg, "hwnd": hwnd, "element": element, "text": text}

    @staticmethod
    def _poll(uia: _Uia, w: dict) -> bool:
        """Read the box once. False when the watch is over."""
        if time.monotonic() > w["deadline"] or user32.GetForegroundWindow() != w["hwnd"]:
            return False
        element = uia.focused()
        if element is None:
            return False
        try:
            if not uia.same(element, w["element"]):
                return False  # focus moved to another box
            text = _text(element)
        finally:
            _release(element)
        if text is None:
            return False

        if w["span"] is None:
            if text == w["text"]:
                w["polls"] += 1
                return w["polls"] < PASTE_POLLS
            start, end = locate(w["text"], text, w["inserted"])
            if not _matches(text[start:end], w["inserted"]):
                w["outcome"] = "the change in the box did not look like the paste"
                return False
            w.update(span=(start, end), last=text, pasted=text[start:end])
            return True

        span = track(w["last"], text, *w["span"])
        if span is None:
            return False  # finish with the last snapshot that still held the span
        w.update(span=span, last=text)
        return True

    def _finish(self, w: dict) -> None:
        _release(w["element"])
        app = w["fg"].exe or "an unknown app"
        if w["span"] is None:
            logger.info(f"Correction capture in {app}: {w['outcome']}.")
            return
        start, end = w["span"]
        final = w["last"][start:end]
        # Unedited stores what was pasted, not the app's copy of it, so a consumer
        # comparing the two never sees CRLF or an expanded mention as a correction.
        source = "unedited" if final == w["pasted"] else "implicit"
        logger.info(f"Correction capture in {app}: {source}.")
        try:
            self._on_final(w["row_id"], w["inserted"] if source == "unedited" else final, source)
        except Exception as exc:
            logger.exception(f"Could not record the final text: {exc}")
