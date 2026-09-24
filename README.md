# Transcriber

Local voice dictation for Windows. Hold a key, speak, and the text appears at your cursor
in whatever application you are using. Everything runs on your machine: no audio and no
text leaves it.

A local alternative to Wispr Flow.

## How it works

Hold **right Ctrl**, speak, release. The audio goes to Faster-Whisper for transcription,
then to a small local LLM through Ollama that removes filler words and spoken
self-corrections and fixes the punctuation, and the result is pasted at your cursor. Your
previous clipboard contents are put back afterwards.

- **Hold** right Ctrl for push-to-talk — recording stops when you let go.
- **Tap** right Ctrl to latch recording on for a long dictation, then tap again to stop.
- **Right Ctrl + Shift** with text selected: speak an instruction such as "make this
  shorter", and the selection is rewritten.

Short dictations skip the LLM entirely, so a two-word reply lands immediately.

The LLM also adapts to where you are typing: terser in a code editor, conversational in
Slack, a bare command in a terminal, full sentences in an email client.

## Requirements

- Windows 10 or 11
- Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU is strongly recommended. Whisper falls back to CPU automatically, but
  dictation then takes tens of seconds rather than a few.
- [Ollama](https://ollama.com/) for the cleanup pass. It is optional — without it you get
  the raw Whisper transcript, which is already punctuated.

## Setup

```powershell
uv sync
ollama pull qwen3:1.7b
uv run python -m app.main
```

The first launch downloads the Whisper model (about 1.5 GB) into the HuggingFace cache.

### Choosing models for your GPU

VRAM is usually the constraint, because Whisper and the refinement model have to fit
alongside each other and alongside the Windows desktop. `bench.py` measures your actual
hardware:

```powershell
uv run python bench.py
```

It records a ten-second sample of your voice, then reports peak VRAM, per-stage latency
and whether Ollama kept the model on the GPU for each combination. Pick the fastest pair
that stays inside your VRAM and reads `GPU` rather than `PARTIAL CPU`, and set it in
Settings.

On a 4 GB card, `distil-large-v3` at `int8_float16` plus a 1.7B–2B refinement model fits.
A 3B model may not.

## The application window

Four screens, reachable from the sidebar. Closing the window leaves the app running in the
tray; quit from the tray menu.

- **History** — every dictation, searchable, with the raw transcript beside what was
  actually pasted, and per-stage timings. Useful when a paste went somewhere unexpected.
- **Stats** — words dictated, speaking rate, estimated time saved, and words per day.
- **Vocabulary** — names, jargon and product terms, fed to Whisper so it spells them
  correctly. Also snippets: say a trigger phrase, get a canned block of text.
- **Settings** — microphone, Whisper model and compute type, Ollama model, the word count
  below which the LLM is skipped, the hold threshold, and autostart at login.

Settings live in `%APPDATA%\transcriber\settings.json`, history in
`%APPDATA%\transcriber\history.db`, logs in `%APPDATA%\transcriber\transcriber.log`.

The tray icon shows the current state: grey idle, red recording, green transcribing, amber
a problem.

## Tests

```powershell
uv run python test_pipeline.py
```

## Known limits

- **English only.** `distil-large-v3` is an English checkpoint. Switch the model to
  `large-v3-turbo` in Settings for other languages, at some cost to English accuracy.
- **Elevated windows.** Windows blocks synthetic keystrokes from a normal process into an
  administrator window, so dictating into an elevated terminal will not work. The app
  detects this and says so rather than failing silently.
- **Right Ctrl is not suppressed.** pynput cannot both suppress a key and report it, so
  right Ctrl still reaches the application underneath. Pressed on its own it does nothing
  in practice.
