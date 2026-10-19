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

## Voice commands

Some phrases turn into what the app in front of you expects typed:

- **"slash ponytail review"** at the start of a dictation, in VS Code or a terminal,
  becomes `/ponytail-review`. This calls a Claude Code skill. Names are matched against
  the skills and commands in `~/.claude`, so spoken words become part of the name when a
  skill by that name exists. Otherwise they become arguments. A slash command skips the
  LLM.
- **"at the rate Rahul"** or **"tag Rahul"** becomes `@Rahul`. **"at the rate app slash
  controller dot py"** becomes `@app/controller.py` (you can also say underscore and
  dash). **"tag everyone"** becomes `@channel` in Slack and `@everyone` in Discord and
  Teams. **"hashtag general"** becomes `#general` in Slack and Discord.
  - These tokens are typed rather than pasted, so the app opens its suggestion popup, and
    Tab picks the match.
  - This works in VS Code (Claude Code's file picker), Slack, Discord and Teams.
  - "tag" works only before a capitalised name, so "tag me later" is left alone.
- **"new line"**, **"new paragraph"** and **"bullet point"** become line breaks and
  `- ` bullets. They are off in code editors, where "add a new line" is ordinary speech,
  and in terminals, where a pasted line break runs the command.

WhatsApp is deliberately left out of mentions. Its popup accepts only with Enter, and
Enter sends the message when nothing matched.

You can set two knobs in `settings.json`:

- `mention_accept_keys` maps an exe name to its accept key (`"tab"` or `"enter"`). Remove
  an app from it to turn mentions off there.
- `mention_popup_ms` is how long to wait for the popup (default 400). Raise it if a slow
  app accepts nothing.

## Install

Download `TranscriberSetup.exe` and double-click it. There are no questions to answer.
Setup installs for your user only, so it needs no administrator rights. Then it:

1. Downloads Python and the app's components into
   `%LOCALAPPDATA%\Programs\Transcriber`. This is up to 3 GB, and more on a PC with an
   NVIDIA GPU, because only that PC gets the CUDA libraries.
2. Tests your hardware. It reads the GPU, VRAM, RAM and CPU, and picks the best models
   they should run. Then it tries them for real: the Windows voice says a test sentence,
   Whisper transcribes it, and the LLM cleans up a sample. A model that is inaccurate,
   too slow, or does not fit on the GPU is dropped for the next smaller one. Ollama is
   installed silently if it is missing. On a PC too small for any LLM, the app pastes the
   raw Whisper transcript, which is already punctuated.
3. Starts Transcriber, and sets it to start at every login, straight to the tray.

The setup is unsigned, so Windows SmartScreen may say "Windows protected your PC". Click
**More info**, then **Run anyway**.

Running setup again updates the app. It keeps your settings and does not rerun the test.
To rerun the test, for example after a GPU upgrade, quit the app from the tray and run:

```powershell
& "$env:LOCALAPPDATA\Programs\Transcriber\.venv\Scripts\transcriber.exe" --setup --force
```

Requirements: Windows 10 or 11, 64-bit, and an internet connection during setup.

## Development

Needs Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --extra cuda      # leave out --extra cuda on a PC without an NVIDIA GPU
ollama pull qwen3:1.7b
uv run python -m app.main
```

The first launch downloads the Whisper model (about 1.5 GB) into the HuggingFace cache.

To build the installer, install [Inno Setup 6](https://jrsoftware.org/isinfo.php)
(`winget install JRSoftware.InnoSetup`), then run `installer\build.ps1`. It writes
`dist\TranscriberSetup.exe`.

### Choosing models by hand

Setup picks the models automatically. To compare them on your own voice, `bench.py`
measures your actual hardware:

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
  **Correct...** records what a dictation should have said. **Fix last dictation...** in
  the tray menu opens it on the latest one.
- **Stats** — words dictated, speaking rate, estimated time saved, and words per day.
- **Vocabulary** — names, jargon and product terms, fed to Whisper so it spells them
  correctly. Also snippets: say a trigger phrase, get a canned block of text.
- **Settings** — microphone, Whisper model and compute type, Ollama model, the word count
  below which the LLM is skipped, the hold threshold, and autostart at login.

Settings live in `%APPDATA%\transcriber\settings.json`, history in
`%APPDATA%\transcriber\history.db`, kept recordings in `%APPDATA%\transcriber\audio\`,
logs in `%APPDATA%\transcriber\transcriber.log`.

### Learning from corrections

After a paste, Transcriber reads the text box you dictated into for up to a minute, through
Windows UI Automation. It follows the pasted text while you edit it. Text you type before
or after the pasted text is ignored. When you leave the box, it records what the pasted
text became. Only that part is stored, never the rest of the box. A correction you make in
History always outranks one read back from the box. Apps that expose no readable text
(some Electron editors, elevated windows) simply record nothing. The log says which apps
could be read.

To turn this off, clear **Learn from my corrections** in Settings. **Keep recordings of
my voice** is off by default. When it is on, each dictation's audio is saved as a WAV
file next to the history.

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
