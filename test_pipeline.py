"""The smallest thing that fails if the non-obvious logic breaks.

    uv run python test_pipeline.py

Plain asserts, no framework. Covers the decisions that are easy to get wrong and hard to
notice: hold-versus-tap, the refinement sanity guard, snippet matching, and the stats
arithmetic, and the clipboard wait. Deliberately does not cover Whisper or Ollama - those
need the real thing, and the manual checklist in the plan covers them.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from app.capture import _matches, locate, track
from app.commands import TOKEN, rewrite, slash_command
from app.controller import normalize_trigger
from app.ui.history import _diff_html
from app.hotkey import Dictation, HoldToggleHotkey
from app.llm import looks_sane
from app.store import SCHEMA, Store, seconds_saved, wpm
from app.transcription import MAX_PROMPT_CHARS, LiveTranscript, build_initial_prompt, tidy
from app.winctx import Foreground, foreground


def _hotkey(threshold_ms: int = 400) -> HoldToggleHotkey:
    return HoldToggleHotkey(on_start=lambda _e: None, on_stop=lambda: None,
                            hold_threshold_ms=threshold_ms)


def test_push_to_talk() -> None:
    """Held past the threshold: start on press, stop on release."""
    k = _hotkey()
    assert k.press(0.0) == Dictation.START
    assert k.release(1.0) == Dictation.STOP
    assert not k.latched


def test_tap_latches_then_stops() -> None:
    """A quick tap keeps recording; the next tap ends it."""
    k = _hotkey()
    assert k.press(0.0) == Dictation.START
    assert k.release(0.1) == Dictation.NOTHING   # short: latch on
    assert k.latched
    assert k.press(2.0) == Dictation.NOTHING     # key down again while latched
    assert k.release(2.1) == Dictation.STOP      # any release while latched stops
    assert not k.latched


def test_long_press_while_latched_still_stops() -> None:
    """Latched, then held a long time: still stops, rather than re-latching."""
    k = _hotkey()
    k.press(0.0)
    k.release(0.1)
    assert k.latched
    k.press(5.0)
    assert k.release(9.0) == Dictation.STOP
    assert not k.latched


def test_autorepeat_does_not_restart() -> None:
    """Windows fires on_press repeatedly while a key is held. Only the first counts,
    otherwise every held key would start a new recording."""
    k = _hotkey()
    assert k.press(0.0) == Dictation.START
    for t in (0.05, 0.10, 0.15, 0.20):
        assert k.press(t) == Dictation.NOTHING
    assert k.release(1.0) == Dictation.STOP


def test_threshold_boundary() -> None:
    """Exactly at the threshold counts as a hold, not a tap."""
    k = _hotkey(threshold_ms=400)
    k.press(0.0)
    assert k.release(0.400) == Dictation.STOP

    k2 = _hotkey(threshold_ms=400)
    k2.press(0.0)
    assert k2.release(0.399) == Dictation.NOTHING
    assert k2.latched


def test_low_threshold_still_latches_a_tap() -> None:
    """A saved threshold of 100 ms made every real tap (90-200 ms) a hold."""
    k = _hotkey(threshold_ms=100)
    k.press(0.0)
    assert k.release(0.15) == Dictation.NOTHING
    assert k.latched


def test_release_without_press_is_ignored() -> None:
    """Right Ctrl released while the app was starting, so the press was never seen."""
    k = _hotkey()
    assert k.release(1.0) == Dictation.NOTHING


def test_reset_clears_latch() -> None:
    k = _hotkey()
    k.press(0.0)
    k.release(0.1)
    assert k.latched
    k.reset()
    assert not k.latched
    assert k.press(2.0) == Dictation.START


def test_sanity_guard_accepts_real_cleanup() -> None:
    raw = "um so i think we should uh ship it on friday you know"
    assert looks_sane(raw, "So I think we should ship it on Friday.")


def test_sanity_guard_accepts_unchanged_text() -> None:
    clean = "The deployment finished at four o'clock."
    assert looks_sane(clean, clean)


def test_sanity_guard_rejects_conversational_openings() -> None:
    """The model talking to the user instead of editing. The single most likely
    failure mode of a small refinement model."""
    raw = "can you tell me what the capital of france is please"
    for bad in [
        "Sure! Here is the edited text: What is the capital of France?",
        "Here's the cleaned up version: what is the capital of France?",
        "The capital of France is Paris.",  # answered instead of edited
        "I'm sorry, I cannot help with that.",
        "Okay, I have removed the filler words.",
        "Note: the text was already clean.",
    ]:
        assert not looks_sane(raw, bad), bad


def test_sanity_guard_rejects_rambling() -> None:
    raw = "send the invoice tomorrow"
    ramble = "send the invoice tomorrow " + "and also consider the following points " * 8
    assert not looks_sane(raw, ramble)


def test_sanity_guard_rejects_summarising() -> None:
    raw = " ".join(["word"] * 40)
    assert not looks_sane(raw, "word word")  # dropped 95%, that is a summary


def test_sanity_guard_rejects_empty() -> None:
    assert not looks_sane("something was said", "")


def test_sanity_guard_rejects_example_leak() -> None:
    """Seen in real use: the prompt's own example pasted in place of the dictation.
    Right length, not one word the speaker said."""
    assert not looks_sane("I see I see I see", "send it Tuesday")


def test_sanity_guard_guards_mentions_and_line_breaks() -> None:
    assert not looks_sane("@Rahul can you check the build today",
                          "Can you check the build today?")
    assert not looks_sane("first point\nsecond point here", "First point, second point here.")
    assert not looks_sane("please rewrite this whole paragraph", "/rewrite this whole paragraph")
    bullets = "Buy these:\n- eggs\n- bread\n- milk"
    assert looks_sane(bullets, bullets)


def _fg(exe: str, context: str) -> Foreground:
    return Foreground(exe, None, context, False)


_APPS = {"code.exe": "tab", "slack.exe": "tab"}


def test_slash_command() -> None:
    """The longest installed name wins, so a skill name beats an argument."""
    skills = {"ponytail", "ponytail-review"}
    assert slash_command("ponytail review", skills) == "/ponytail-review"
    assert slash_command("ponytail ultra", skills) == "/ponytail ultra"
    assert slash_command("Pony tail ultra.", skills) == "/ponytail ultra"  # Whisper split it
    assert slash_command("clear.", skills) == "/clear"  # not installed: taken literally

    code = _fg("Code.exe", "code")
    assert rewrite("Slash ponytail review.", code, _APPS, skills) == "/ponytail-review"
    assert rewrite("Slash ponytail review.", _fg("slack.exe", "chat"), _APPS, skills) \
        == "Slash ponytail review."
    # A second command becomes one only when it names a known skill.
    skills = {"ponytail", "caveman"}
    assert rewrite("slash ponytail ultra slash caveman ultra", code, _APPS, skills) \
        == "/ponytail ultra /caveman ultra"
    assert rewrite("Slash ponytail ultra. Slash caveman ultra.", code, _APPS, skills) \
        == "/ponytail ultra /caveman ultra"
    assert rewrite("slash ponytail ultra slash foo", code, _APPS, skills) \
        == "/ponytail ultra slash foo"


def test_tidy() -> None:
    assert tidy("such as such as the ability to directly directly use") \
        == "such as the ability to directly use"
    assert tidy("because of the because of the whisper model") == "because of the whisper model"
    assert tidy("will will  Will the LLM move") == "will the LLM move"
    assert tidy("um so I think uh, we ship") == "so I think we ship"
    for untouched in ["Yes. Yes.", "I think this is it", "10:30 tomorrow"]:
        assert tidy(untouched) == untouched


def test_rewrite_mentions() -> None:
    slack, code = _fg("slack.exe", "chat"), _fg("Code.exe", "code")
    assert rewrite("At the rate Rahul, can you check this?", slack, _APPS) \
        == "@Rahul, can you check this?"
    assert rewrite("Tag everyone, standup moved to 5.", slack, _APPS) \
        == "@channel, standup moved to 5."
    assert rewrite("Ask tag Priya about it.", slack, _APPS) == "Ask @Priya about it."
    assert rewrite("Post it in hashtag general.", slack, _APPS) == "Post it in #general."
    assert rewrite("Explain at the rate app slash controller dot py.", code, _APPS, set()) \
        == "Explain @app/controller.py."
    assert rewrite("add the rate readme dot md", code, _APPS, set()) == "@readme.md"
    for untouched in ["Prices rose at the rate of 5 percent.", "I'll tag you later.",
                      "The price tag is too high."]:
        assert rewrite(untouched, slack, _APPS) == untouched
    # Mentions only in apps whose popup can be driven; WhatsApp is not one of them.
    said = "At the rate Rahul, can you check this?"
    assert rewrite(said, _fg("WhatsApp.exe", "chat"), _APPS) == said


def test_rewrite_formatting() -> None:
    chat = _fg("slack.exe", "chat")
    assert rewrite("first thing new line second thing", chat, {}) == "first thing\nsecond thing"
    assert rewrite("Intro. New paragraph. Details.", chat, {}) == "Intro.\n\nDetails."
    assert rewrite("Buy bullet point eggs bullet point bread", chat, {}) == "Buy\n- eggs\n- bread"
    # In a prompt to a coding agent "new line" is just words.
    said = "Add a new line after the import."
    assert rewrite(said, _fg("Code.exe", "code"), {}, set()) == said


def test_mention_tokens() -> None:
    assert TOKEN.split("hi @Rahul, see #general.") == ["hi ", "@Rahul", ", see ", "#general", "."]
    assert TOKEN.findall("mail me@example.com about C# and @app/controller.py.") \
        == ["@app/controller.py"]


def test_snippet_normalisation() -> None:
    """Whisper punctuates and capitalises, so a trigger must match through both."""
    assert normalize_trigger("My address.") == "my address"
    assert normalize_trigger("  my  address  ") == "my  address"
    assert normalize_trigger("Sign off!") == "sign off"
    assert normalize_trigger('"quoted"') == "quoted"
    assert normalize_trigger("My address?") == normalize_trigger("my address")


def test_wpm_and_time_saved() -> None:
    assert wpm(100, 60.0) == 100.0
    assert wpm(50, 30.0) == 100.0
    assert wpm(10, 0.0) == 0.0  # no division by zero on a zero-length recording

    # 40 words at the 40 WPM typing baseline is 60s of typing, against 10s spoken.
    assert abs(seconds_saved(40, 10.0) - 50.0) < 1e-9


def test_time_saved_never_negative() -> None:
    """Rambling slowly saves nothing. It must not report a negative."""
    assert seconds_saved(2, 60.0) == 0.0
    assert seconds_saved(0, 30.0) == 0.0


def test_initial_prompt() -> None:
    assert build_initial_prompt([]) is None
    assert build_initial_prompt(["   ", ""]) is None
    assert build_initial_prompt(["Kubernetes", "ctranslate2"]) == "Kubernetes, ctranslate2"

    # Overflow truncates rather than dropping the lot or exceeding the prompt window.
    long_list = [f"term{i:04d}" for i in range(400)]
    prompt = build_initial_prompt(long_list)
    assert prompt is not None
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert prompt.startswith("term0000, term0001")


def test_store_roundtrip_and_stats() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "history.db")
        store.add(raw="um hello there friend", refined="Hello there, friend.",
                  audio_s=4.0, app_exe="Code.exe", app_title="t", context="code",
                  asr_ms=300, llm_ms=400)
        store.add(raw="yes", refined=None, audio_s=1.0,
                  app_exe="slack.exe", app_title="t", context="chat", asr_ms=120)

        rows = store.recent()
        assert len(rows) == 2

        stats = store.stats()
        assert stats.count == 2
        # Words are counted on the text actually pasted: 3 refined + 1 raw.
        assert stats.words == 4
        assert stats.llm_share == 0.5
        assert stats.speaking_wpm > 0

        row_id = int(rows[0]["id"])
        store.delete(row_id)
        assert len(store.recent()) == 1
        store.close()


def test_store_counts_words_of_pasted_text() -> None:
    """The refined text is what landed in the document, so it is what stats must count."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "h.db")
        store.add(raw="um uh so like basically yes", refined="Yes.", audio_s=3.0,
                  app_exe=None, app_title=None, context=None)
        assert store.stats().words == 1
        store.close()


def test_daily_words_zero_fills_quiet_days() -> None:
    """The chart's x-axis is calendar days, so days with no dictation must be present."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "h.db")
        store.add(raw="one two three", refined=None, audio_s=1.0,
                  app_exe=None, app_title=None, context=None)
        daily = store.daily_words(days=3)
        assert [words for _day, words in daily] == [0, 0, 3], daily
        assert daily[-1][0] == str(date.today())
        store.close()


def test_diff_marks_llm_edits_not_case_or_punctuation() -> None:
    html = _diff_html("um send it tuesday", "Send it on Tuesday.", "DIM", "ACC")
    assert '<s style="color:DIM">um</s>' in html, html
    assert '<span style="color:ACC">on</span>' in html, html
    # Capitalising and punctuating are not edits worth highlighting.
    assert "Send it" in html and "Tuesday." in html and html.count("<s ") == 1, html


def test_foreground_never_raises() -> None:
    """Whatever has focus, and whatever its integrity level, this must return."""
    fg = foreground()
    assert fg.context in {"code", "chat", "email", "terminal", "prose"}
    assert isinstance(fg.elevated_guess, bool)


# A stand-in for the app receiving Ctrl+V: it holds the clipboard open while it reads.
_PASTING_APP = r"""
import ctypes
u = ctypes.windll.user32; k = ctypes.windll.kernel32
u.GetClipboardData.restype = ctypes.c_void_p
k.GlobalLock.restype = ctypes.c_wchar_p; k.GlobalLock.argtypes = [ctypes.c_void_p]
while not u.OpenClipboard(None): pass
print("open", flush=True)
h = u.GetClipboardData(13)
print(k.GlobalLock(h) if h else None, flush=True)
u.CloseClipboard()
"""


def test_clipboard_restore_while_app_reads() -> None:
    """The settle wait must answer the reader, or it stalls and the restore fails.

    Uses the real clipboard, and puts back what was on it.
    """
    import subprocess
    import sys

    from PySide6.QtGui import QGuiApplication

    from app import output

    app = QGuiApplication.instance() or QGuiApplication([])
    clipboard = app.clipboard()
    before = output._snapshot()
    try:
        clipboard.setText("dictated")
        reader = subprocess.Popen([sys.executable, "-c", _PASTING_APP],
                                  stdout=subprocess.PIPE, text=True)
        reader.stdout.readline()  # the reader has the clipboard open
        output._wait(150)
        clipboard.setText("restored")
        assert clipboard.ownsClipboard(), "restore failed: the reader still holds the clipboard"
        while reader.poll() is None:
            output._wait(10)
        assert reader.stdout.read().strip() == "dictated", "the reader did not get the dictation"
    finally:
        if before is not None:
            clipboard.setMimeData(before)


def test_unusable_mic_falls_back_to_default() -> None:
    import sounddevice as sd
    from app.audio import AudioRecorder

    def check(device=None, **_kw):
        if device == 9:  # a WASAPI mic: native 48 kHz only
            raise sd.PortAudioError("Invalid sample rate", -9997)

    opened = []

    class FakeStream:
        def __init__(self, device=None, **_kw):
            opened.append(device)

        def start(self):
            pass

    real = sd.check_input_settings, sd.InputStream
    sd.check_input_settings, sd.InputStream = check, FakeStream
    try:
        AudioRecorder(device_index=9).start()
    finally:
        sd.check_input_settings, sd.InputStream = real
    assert opened == [None], opened


def test_recorder_level() -> None:
    import numpy as np
    from app.audio import AudioRecorder

    recorder = AudioRecorder()
    recorder._callback(np.full((1024, 1), 0.5, np.float32), 1024, None, None)
    assert abs(recorder.level - 0.5) < 1e-6, recorder.level


class _SizeWhisper:
    """Stands in for Whisper: 'transcribes' a segment as its length in VAD windows."""

    def transcribe(self, audio, _prompt):
        return str(audio.size // 512)


def test_live_transcript_cuts_at_pauses() -> None:
    """A pause after speech ends a segment; silence before speech does not; the tail
    goes out on finish, in order."""
    live = LiveTranscript(_SizeWhisper(), None, pause_ms=96)  # 3 windows of 512
    live._score = lambda w: (np.abs(w).max(axis=1) > 0.5).astype(np.float32)
    speech, quiet = np.ones((1024, 1), np.float32), np.zeros((1024, 1), np.float32)
    for block in (speech, speech, quiet, quiet, speech):
        live.push(block)
    # 4 speech + 3 quiet windows cut; then 1 quiet + 2 speech left for the tail.
    assert live.finish() == "7 3"


def test_vad_scores_silence_low() -> None:
    """The real Silero call still takes and returns what we assume across upgrades."""
    live = LiveTranscript(_SizeWhisper(), None, pause_ms=700)
    probs = live._score(np.zeros((3, 512), np.float32))
    live.close()
    assert probs.shape == (3,) and (probs < 0.5).all(), probs


def test_locate_insertion() -> None:
    assert locate("", "Hello there.", "Hello there.") == (0, 12)
    before, pasted = "Hi Sam. Thanks.", "See you at 10. "
    after = before.replace("Thanks.", pasted + "Thanks.")
    start, end = locate(before, after, pasted)
    assert after[start:end] == pasted, after[start:end]
    # Pasted over a similar selection: the bare difference is one "r", not the paste.
    start, end = locate("Send it tomorow please", "Send it tomorrow please", "tomorrow")
    assert "Send it tomorrow please"[start:end] == "tomorrow"
    # The app changed the paste (an accepted mention): the box's version is the span.
    after = "@Rahul Sharma can you check"
    start, end = locate("", after, "@Rahul can you check")
    assert after[start:end] == after and _matches(after, "@Rahul can you check")
    assert not _matches("something else entirely", "@Rahul can you check")


def test_track_region() -> None:
    """Edits inside the pasted span are the correction; typing at its edges is not."""
    box = "Hi Sam. See you at ten. Thanks."
    span = (box.index("See"), box.index("Thanks."))

    fixed = box.replace("ten", "10")
    start, end = track(box, fixed, *span)
    assert fixed[start:end] == "See you at 10. "

    typed_after = box.replace("Thanks.", "Bring the doc. Thanks.")
    start, end = track(box, typed_after, *span)
    assert typed_after[start:end] == "See you at ten. "

    typed_before = "Hello! " + box
    start, end = track(box, typed_before, *span)
    assert typed_before[start:end] == "See you at ten. "

    assert track(box, "", *span) is None                                # sent, or cleared
    assert track(box, box.replace("Sam. See", "Sam, see"), *span) is None  # crosses an edge

    # The case that matters most: fix the last word, then keep typing, polls apart.
    prev, span = "Hello wrold", (0, 11)
    for cur in ("Hello world", "Hello world. And more"):
        span = track(prev, cur, *span)
        prev = cur
    assert prev[span[0]:span[1]] == "Hello world"


def test_store_migrates_old_schema() -> None:
    """A history.db from before corrections keeps its rows and gains the new columns."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "h.db"
        old = sqlite3.connect(path)
        old.executescript(SCHEMA)
        old.execute("INSERT INTO dictations (ts, raw, audio_s, words) "
                    "VALUES ('2026-01-01T00:00:00+00:00', 'hi', 1.0, 1)")
        old.commit()
        old.close()

        store = Store(path)
        [row] = store.recent()
        assert row["raw"] == "hi" and row["final"] is None and row["is_edit"] == 0
        store.add(raw="make it formal", refined="Dear Sam,", audio_s=1.0, app_exe=None,
                  app_title=None, context=None, is_edit=True)
        assert store.recent()[0]["is_edit"] == 1
        store.close()
        Store(path).close()  # opening an already migrated database is a no-op


def test_explicit_correction_beats_implicit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "h.db")
        row_id = store.add(raw="send it tuesday", refined="Send it Tuesday.", audio_s=1.0,
                           app_exe=None, app_title=None, context=None)
        store.set_final(row_id, "Send it Thursday.", "explicit")
        store.set_final(row_id, "Send it Tuesday.", "unedited")  # read back later
        [correction] = store.corrections()
        assert correction["final"] == "Send it Thursday.", correction
        assert correction["weight"] == 1.0
        store.set_final(row_id, "Send it on Thursday.", "explicit")  # a later fix wins
        assert store.corrections()[0]["final"] == "Send it on Thursday."
        store.close()


def test_store_audio_roundtrip() -> None:
    import wave

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "h.db")
        row_id = store.add(raw="hi", refined=None, audio_s=0.5, app_exe=None,
                           app_title=None, context=None)
        store.save_audio(row_id, np.full(8000, 0.5, np.float32), 16000)
        with wave.open(str(store.audio_path(row_id))) as w:
            assert (w.getnframes(), w.getframerate(), w.getsampwidth()) == (8000, 16000, 2)
        store.delete(row_id)
        assert not store.audio_path(row_id).exists()
        store.close()


def test_field_read_never_raises() -> None:
    """Whatever has focus, reading it back gives text or None. On its own thread, as in
    the app: the clipboard test has made the main thread Qt's."""
    import threading

    from app import capture

    out = []

    def read() -> None:
        uia = capture._Uia()
        element = uia.focused()
        out.append(capture._text(element) if element else None)
        capture._release(element)

    thread = threading.Thread(target=read)
    thread.start()
    thread.join(10)
    assert out and (out[0] is None or isinstance(out[0], str)), out


def main() -> int:
    tests =[v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for test in tests:
        try:
            test()
            print(f"  pass  {test.__name__}")
        except AssertionError as exc:
            failed.append(test.__name__)
            print(f"  FAIL  {test.__name__}: {exc or 'assertion failed'}")
        except Exception as exc:
            failed.append(test.__name__)
            print(f"  ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
