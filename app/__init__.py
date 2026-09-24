"""Local voice dictation for Windows.

Modules:
- config: settings dataclasses, plus load and save.
- hotkey: right Ctrl as push-to-talk and toggle.
- audio: microphone capture.
- transcription: Faster-Whisper wrapper.
- llm: Ollama client for cleanup and voice editing.
- output: clipboard save/restore and paste at the cursor.
- winctx: which application has focus.
- store: SQLite history and stats.
- controller: the dictation pipeline.
- autostart: launch at login.
- ui: the application window, its screens, and the tray icon.
- main: entry point.
"""

# Must run before anything imports ctranslate2, which faster_whisper does at import
# time. Importing app.* therefore always registers the CUDA DLL directories first.
from . import cuda_dlls as _cuda_dlls  # noqa: E402

_cuda_dlls.register()
