"""The hardware test the installer runs: `transcriber.exe --setup`.

Picks the Whisper and Ollama models this PC can actually run and writes them to
settings.json, so a fresh install works without anyone opening Settings.

Two halves. `candidates()` turns what the hardware reports into tiers, best first, from a
fixed table. Then each tier is tried for real: a sentence spoken by the Windows voice is
transcribed, and one cleanup goes through the LLM. The first tier that is accurate, fast
enough and fits in VRAM wins. The table only has to be roughly right; the trial is what
catches a model that spills to the CPU or is too slow on this particular machine.

Skipped when settings.json already exists, so rerunning the installer keeps the user's
own choices. `--setup --force` runs it anyway.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
from loguru import logger
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QWidget

from . import autostart, config
from .config import OllamaConfig, WhisperConfig
from .llm import OLLAMA_EXE, OllamaClient
from .transcription import SAMPLE_RATE, TranscriptionService
from .ui import theme
from .ui.settings import ollama_models

# ponytail: a fixed table, rough on purpose. The trial run is what catches a tier that is
# too slow or does not fit; these thresholds are the calibration knobs.
GPU_LARGE_MIB = 7000   # float16 Whisper, with room for a 4B model beside it
GPU_MIN_MIB = 3500     # a "4 GB" card reports 4096; below this Whisper and an LLM do not share
RAM_MIN_MIB = 7500     # "8 GB" reports a little under; below it a CPU LLM starves Windows
CPU_TIERS = [("small.en", "cpu", "int8"), ("base.en", "cpu", "int8")]

WHISPER_MAX_RTF = 0.5  # seconds spent transcribing per second of speech
MIN_HEARD = 0.8        # share of the test sentence's words Whisper must get right
LLM_MAX_MS = 3000      # one cleanup, model already loaded

PHRASE = "Please move the design review to Thursday afternoon and send the notes to the whole team."
LLM_SAMPLE = "um so I think we should uh move the the planning meeting to friday no wait monday morning"
OLLAMA_SETUP_URL = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_SIGNER = "O=Ollama Inc."
_NO_WINDOW = subprocess.CREATE_NO_WINDOW
_WORDS = re.compile(r"[a-z]+")

Say = Callable[[str], None]
Progress = Callable[[int, int], None]


@dataclass
class Hardware:
    gpu: Optional[str]  # NVIDIA GPU name; None without one
    vram_mib: int
    cuda: bool          # ctranslate2 can run on it
    ram_mib: int
    cpus: int

    def __str__(self) -> str:
        gpu = f"{self.gpu} ({self.vram_mib / 1024:.0f} GB)" if self.gpu else "no NVIDIA GPU"
        return f"{gpu}, {self.ram_mib / 1024:.0f} GB RAM, {self.cpus} CPU threads"


# --- what the hardware is ---

def _nvidia() -> tuple[Optional[str], int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True, creationflags=_NO_WINDOW,
        ).stdout
        name, mib = out.strip().splitlines()[0].rsplit(",", 1)
        return name.strip(), int(mib)
    except Exception:
        return None, 0


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")
    ]


def _ram_mib() -> int:
    status = _MemoryStatus(dwLength=ctypes.sizeof(_MemoryStatus))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullTotalPhys // 2**20


def _cuda_usable() -> bool:
    """ctranslate2 counts the GPU even when cuBLAS is missing, and the first transcription
    then hung rather than raising (seen without the cuda extra). So load what it needs."""
    try:
        import ctranslate2  # noqa: PLC0415 - after app/__init__ registered the CUDA DLLs

        if ctranslate2.get_cuda_device_count() < 1:
            return False
        ctypes.WinDLL("cublas64_12.dll")
        ctypes.WinDLL("cudnn64_9.dll")
        return True
    except Exception:
        return False


def probe() -> Hardware:
    gpu, vram = _nvidia()
    return Hardware(gpu, vram, _cuda_usable(), _ram_mib(), os.cpu_count() or 1)


def candidates(hw: Hardware) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Whisper (model, device, compute_type) tiers and Ollama model tiers, best first.
    An empty LLM list means the PC is too small for one."""
    if hw.cuda and hw.vram_mib >= GPU_LARGE_MIB:
        return ([("distil-large-v3", "cuda", "float16"),
                 ("distil-large-v3", "cuda", "int8_float16")] + CPU_TIERS,
                ["qwen3:4b", "qwen3:1.7b"])
    # No qwen3:0.6b tier: in the trial it turned "friday no wait monday morning" into
    # "Friday, 10:00 AM". An LLM that changes the meaning is worse than none.
    if hw.cuda and hw.vram_mib >= GPU_MIN_MIB:
        return [("distil-large-v3", "cuda", "int8_float16")] + CPU_TIERS, ["qwen3:1.7b"]
    if hw.ram_mib >= RAM_MIN_MIB:
        return CPU_TIERS, ["qwen3:1.7b"]
    return CPU_TIERS, []


# --- Whisper trial ---

def speak(path: Path) -> Optional[np.ndarray]:
    """The Windows voice says PHRASE; returns it as 16 kHz mono float32, or None."""
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        " [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,"
        " [System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{str(path).replace(chr(39), chr(39) * 2)}', $f);"
        f"$s.Speak('{PHRASE}'); $s.Dispose()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, timeout=60, check=True, creationflags=_NO_WINDOW)
        with wave.open(str(path), "rb") as w:
            frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as exc:
        logger.warning(f"No Windows voice for the test sentence ({exc}); checking model loads only.")
        return None


def heard(text: str) -> float:
    """Share of PHRASE's words that appear in `text`."""
    expected, got = _WORDS.findall(PHRASE.lower()), set(_WORDS.findall(text.lower()))
    return sum(w in got for w in expected) / len(expected)


def pick_whisper(tiers: list[tuple[str, str, str]], audio: Optional[np.ndarray],
                 say: Say) -> tuple[tuple[str, str, str], TranscriptionService]:
    """First tier that loads on the device it asked for, hears the phrase, and keeps up.
    The last tier is kept even when slow: something beats nothing."""
    for i, (model, device, compute) in enumerate(tiers):
        say(f"Testing speech model {model} on the {'GPU' if device == 'cuda' else 'CPU'}. "
            "The first use downloads it.")
        try:
            service = TranscriptionService(WhisperConfig(model_size=model, device=device,
                                                         compute_type=compute))
            service.warmup()
            if audio is None:
                # ponytail: without a voice only loading is proven; transcribe() still
                # falls back to CPU on the first dictation if CUDA fails later.
                return (model, device, compute), service
            t0 = time.perf_counter()
            text = service.transcribe(audio) or ""
            rtf = (time.perf_counter() - t0) / (audio.size / SAMPLE_RATE)
        except Exception as exc:
            logger.warning(f"Whisper {model} {device} {compute} failed: {exc}")
            continue
        score = heard(text)
        logger.info(f"Whisper {model} {device} {compute}: rtf {rtf:.2f}, heard {score:.0%}: {text!r}")
        # service.device differs when transcribe() fell back from CUDA to CPU.
        if (service.device == device and score >= MIN_HEARD and rtf <= WHISPER_MAX_RTF) \
                or i == len(tiers) - 1:
            return (model, device, compute), service
        del service
    raise RuntimeError("No speech model could be loaded.")


# --- Ollama ---

def _download(url: str, path: Path, progress: Progress) -> None:
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total, done = int(resp.headers.get("content-length", 0)), 0
        with open(path, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                progress(done, total)


def _install_ollama(folder: Path, say: Say, progress: Progress) -> None:
    say("Downloading Ollama, which runs the language model.")
    setup = folder / "OllamaSetup.exe"
    _download(OLLAMA_SETUP_URL, setup, progress)
    # Never run a downloaded installer without checking who signed it.
    signer = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         f"$s = Get-AuthenticodeSignature -LiteralPath '{str(setup).replace(chr(39), chr(39) * 2)}';"
         " if ($s.Status -eq 'Valid') { $s.SignerCertificate.Subject }"],
        capture_output=True, text=True, timeout=120, creationflags=_NO_WINDOW,
    ).stdout
    if OLLAMA_SIGNER not in signer:
        raise RuntimeError(f"OllamaSetup.exe is not validly signed by Ollama ({signer.strip() or 'no signature'}).")
    say("Installing Ollama.")
    subprocess.run([str(setup), "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
                   check=True, timeout=1800)


def ensure_ollama(client: OllamaClient, folder: Path, say: Say, progress: Progress) -> bool:
    """Ollama reachable, starting or installing it first when needed."""
    if client.is_available(force=True):  # also starts an installed but stopped Ollama
        return True
    if not (OLLAMA_EXE.exists() or shutil.which("ollama")):
        _install_ollama(folder, say, progress)
    say("Starting Ollama.")
    for _ in range(30):
        if client.is_available(force=True):
            return True
        time.sleep(1)
    return False


def residency(base_url: str, model: str) -> str:
    """GPU, PARTIAL CPU or unknown - the only reliable way to catch a silent spill."""
    try:
        for m in requests.get(f"{base_url}/api/ps", timeout=5).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                total, vram = m.get("size", 0), m.get("size_vram", 0)
                if not total:
                    return "unknown"
                if vram >= total:
                    return "GPU"
                return f"PARTIAL CPU ({vram / total:.0%} on GPU)"
    except Exception:
        pass
    return "unknown"


def _pull(base_url: str, model: str, progress: Progress) -> None:
    with requests.post(f"{base_url}/api/pull", json={"model": model, "stream": True},
                       stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            event = json.loads(line) if line else {}
            if "error" in event:
                raise RuntimeError(event["error"])
            progress(event.get("completed", 0), event.get("total", 0))


def pick_llm(tiers: list[str], base: OllamaConfig, need_gpu: bool,
             say: Say, progress: Progress) -> Optional[str]:
    """First model that cleans up the sample sanely, fast, and (on NVIDIA) fully on GPU."""
    have = set(ollama_models(base.base_url))
    for model in tiers:
        pulled = model not in have
        client = OllamaClient(replace(base, model=model, enabled=True))
        try:
            if pulled:
                say(f"Downloading language model {model}.")
                _pull(base.base_url, model, progress)
            say(f"Testing language model {model}.")
            client.warm()
            text, ms = client.refine(LLM_SAMPLE)  # ms is None when the request failed
            where = residency(base.base_url, model)
            logger.info(f"LLM {model}: {ms}ms, {where}: {text!r}")
            # "monday": the sample's self-correction must survive, the meaning intact.
            if ms is not None and ms <= LLM_MAX_MS and "monday" in text.lower() \
                    and (where == "GPU" or not need_gpu):
                return model
            requests.post(f"{base.base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
            if pulled:  # do not leave gigabytes behind for a model nobody will use
                requests.delete(f"{base.base_url}/api/delete", json={"model": model}, timeout=30)
        except Exception as exc:
            logger.warning(f"LLM {model} failed: {exc}")
        finally:
            client.close()
    return None


# --- the whole test ---

def configure(say: Say, progress: Progress) -> str:
    """Probe, trial, save. Returns the line shown when done."""
    hw = probe()
    logger.info(f"Hardware: {hw}, cuda={hw.cuda}")
    say(f"Found {hw}.")
    whisper_tiers, llm_tiers = candidates(hw)
    cfg = config.load()  # defaults, or the user's own file under --force

    with tempfile.TemporaryDirectory() as tmp:
        say("Making a test recording with the Windows voice.")
        audio = speak(Path(tmp) / "test.wav")
        (model, device, compute), service = pick_whisper(whisper_tiers, audio, say)
        cfg.whisper.model_size, cfg.whisper.device, cfg.whisper.compute_type = model, device, compute

        llm = None
        if llm_tiers:
            say("Checking for Ollama, which runs the language model.")
            client = OllamaClient(cfg.ollama)
            try:
                if ensure_ollama(client, Path(tmp), say, progress):
                    need_gpu = hw.cuda and hw.vram_mib >= GPU_MIN_MIB
                    llm = pick_llm(llm_tiers, cfg.ollama, need_gpu, say, progress)
            except Exception as exc:
                logger.warning(f"Ollama setup failed; continuing without the LLM: {exc}")
            finally:
                client.close()
        del service

    cfg.ollama.enabled = llm is not None
    if llm:
        cfg.ollama.model = llm
    # The intent, not the registry: security software can deny setup's write, and
    # main.py then creates the entry on a later launch.
    cfg.autostart = True
    cfg.save()
    where = "GPU" if device == "cuda" else "CPU"
    cleanup = f" with {llm} cleanup" if llm else " (no LLM cleanup on this PC)"
    return f"Ready: {model} on the {where}{cleanup}. Hold right Ctrl to dictate."


class _Bridge(QObject):
    """Worker-thread emits, queued onto the GUI thread."""
    message = Signal(str)
    progress = Signal(int, int)
    finished = Signal(str, bool)


class SetupWindow(QWidget):
    def __init__(self) -> None:
        # On top, so the installer's own window cannot hide the only visible progress.
        super().__init__(None, Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Transcriber setup")
        self.setFixedWidth(520)
        self._status = theme.label("Checking this PC's hardware.", wrap=True)
        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setRange(0, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        layout.addWidget(theme.label("Setting up Transcriber for this PC", "SectionTitle"))
        layout.addWidget(self._status)
        layout.addWidget(self._bar)
        layout.addWidget(theme.label(
            "This picks the best speech and language models your hardware can run, and "
            "downloads them. It takes a few minutes. There is nothing to do but wait.",
            "Caption", wrap=True))

    def set_message(self, text: str) -> None:
        self._status.setText(text)
        self._bar.setRange(0, 0)  # busy, until a download reports how far it is

    def set_progress(self, done: int, total: int) -> None:
        if total:
            self._bar.setRange(0, 1000)  # bytes overflow a QProgressBar's int
            self._bar.setValue(int(done * 1000 / total))

    def finish(self, text: str) -> None:
        self._status.setText(text)
        self._bar.setRange(0, 1)
        self._bar.setValue(1)


def run(app: QApplication, force: bool = False) -> int:
    """The installer's step. 0 when configured or skipped, 1 when defaults were kept."""
    if config.SETTINGS_PATH.exists() and not force:
        logger.info("Settings exist; skipping the hardware test. Use --setup --force to rerun it.")
        return 0

    window, bridge, result = SetupWindow(), _Bridge(), [1]
    bridge.message.connect(window.set_message)
    bridge.progress.connect(window.set_progress)

    def done(text: str, ok: bool) -> None:
        window.finish(text)
        result[0] = 0 if ok else 1
        QTimer.singleShot(3000 if ok else 10000, app.quit)  # a failure needs time to read

    bridge.finished.connect(done)

    def work() -> None:
        # First, so the app starts at login even when the test fails. Here, off the GUI
        # thread, because security software can hold this write for half a minute.
        autostart.set_enabled(True)
        try:
            bridge.finished.emit(configure(bridge.message.emit, bridge.progress.emit), True)
        except Exception as exc:
            logger.exception(f"Hardware test failed: {exc}")
            bridge.finished.emit(f"The hardware test failed ({exc}). Transcriber will "
                                 "start with its default settings.", False)

    window.show()
    threading.Thread(target=work, daemon=True).start()
    app.exec()
    return result[0]
