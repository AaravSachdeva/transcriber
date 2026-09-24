"""Phase 1 benchmark: does a Whisper + Ollama pair fit in 4 GB of VRAM, and how slow is it?

Records one real speech sample from your mic on first run and reuses it for every
combination, so the numbers are comparable and reflect your actual voice and microphone.

    uv run python bench.py
    uv run python bench.py --record   # discard the cached sample and record a new one
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests

SAMPLE_PATH = Path("bench_sample.wav")
SAMPLE_RATE = 16_000
RECORD_SECONDS = 10
# 127.0.0.1, not localhost: on Windows the latter resolves to ::1 first and Ollama
# listens on IPv4 only, adding roughly 2.2 seconds to every request.
OLLAMA = "http://127.0.0.1:11434"
SESSION = requests.Session()

COMPUTE_TYPES = ["int8_float16", "int8"]
LLM_MODELS = ["llama3.2:3b", "gemma2:2b", "qwen3:1.7b"]

# Matches the prompt the real app will use, so the token count is representative.
SYSTEM_PROMPT = (
    "Rewrite the user's dictated text as clean prose. Fix punctuation and "
    "capitalisation. Remove filler words and spoken self-corrections. Output only "
    "the rewritten text."
)


def nvidia_used_mib() -> int | None:
    """Total VRAM in use on GPU 0, across all processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VramPeak:
    """Polls nvidia-smi in the background and keeps the highest reading."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak = 0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VramPeak":
        def poll() -> None:
            while not self._stop.is_set():
                used = nvidia_used_mib()
                if used is not None:
                    self.peak = max(self.peak, used)
                self._stop.wait(self._interval)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def record_sample() -> None:
    import sounddevice as sd

    print(f"\nRecording {RECORD_SECONDS}s. Speak normally, as you would when dictating.")
    for n in (3, 2, 1):
        print(f"  {n}...", flush=True)
        time.sleep(1)
    print("  GO", flush=True)
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("  done")

    pcm16 = (np.clip(audio[:, 0], -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(SAMPLE_PATH), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())
    print(f"  saved {SAMPLE_PATH} ({SAMPLE_PATH.stat().st_size / 1024:.0f} KiB)")


def load_sample() -> np.ndarray:
    with wave.open(str(SAMPLE_PATH), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"sample is {w.getframerate()} Hz"
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def ollama_running() -> bool:
    try:
        return SESSION.get(f"{OLLAMA}/api/tags", timeout=3).ok
    except Exception:
        return False


def installed_models() -> set[str]:
    try:
        tags = SESSION.get(f"{OLLAMA}/api/tags", timeout=5).json()
        return {m["name"] for m in tags.get("models", [])}
    except Exception:
        return set()


def unload(model: str) -> None:
    """keep_alive 0 evicts the model immediately, so the next combo starts clean."""
    try:
        SESSION.post(f"{OLLAMA}/api/chat",
                     json={"model": model, "messages": [], "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def ollama_residency(model: str) -> str:
    """GPU, PARTIAL CPU or unknown - the only reliable way to catch a silent spill."""
    try:
        for m in SESSION.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models", []):
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


def refine(model: str, text: str) -> tuple[str, int]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": text}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 1024, "num_predict": 512},
    }
    # qwen3 is a hybrid-thinking model. Left on, it spends hundreds of tokens
    # reasoning before answering, which lands squarely in the dictation latency path.
    if model.startswith("qwen3"):
        body["think"] = False
    t0 = time.perf_counter()
    resp = SESSION.post(f"{OLLAMA}/api/chat", json=body, timeout=180)
    resp.raise_for_status()
    ms = int((time.perf_counter() - t0) * 1000)
    return (resp.json().get("message", {}).get("content") or "").strip(), ms


def bench_pair(audio: np.ndarray, compute_type: str, llm: str) -> dict:
    from faster_whisper import WhisperModel

    row = {"compute_type": compute_type, "llm": llm}
    with VramPeak() as vram:
        baseline = nvidia_used_mib() or 0

        t0 = time.perf_counter()
        model = WhisperModel("distil-large-v3", device="cuda", compute_type=compute_type)
        # Warm the kernels so load cost does not land on the measured run.
        list(model.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
                              language="en", beam_size=1)[0])
        row["load_ms"] = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            audio, language="en", beam_size=5,
            condition_on_previous_text=False, vad_filter=True, without_timestamps=True,
        )
        text = " ".join(s.text for s in segments).strip()
        row["asr_ms"] = int((time.perf_counter() - t0) * 1000)
        row["words"] = len(text.split())

        try:
            refined, row["llm_ms"] = refine(llm, text)
            row["residency"] = ollama_residency(llm)
            row["refined_words"] = len(refined.split())
        except Exception as exc:
            row["llm_ms"] = -1
            row["residency"] = f"FAILED: {type(exc).__name__}"
            row["refined_words"] = 0

        row["peak_mib"] = vram.peak
        row["baseline_mib"] = baseline

    del model
    unload(llm)
    time.sleep(2)  # let the driver actually release the allocation before the next combo
    row["total_ms"] = row["asr_ms"] + max(row["llm_ms"], 0)
    row["text"] = text
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="record a fresh sample")
    args = ap.parse_args()

    if args.record or not SAMPLE_PATH.exists():
        record_sample()
    audio = load_sample()
    print(f"\nSample: {audio.size / SAMPLE_RATE:.1f}s at {SAMPLE_RATE} Hz")

    if not ollama_running():
        print("\nOllama is not reachable on localhost:11434. Start it, then re-run.")
        return 1

    have = installed_models()
    todo = []
    for llm in LLM_MODELS:
        if llm in have or f"{llm}:latest" in have:
            todo.append(llm)
        else:
            print(f"  skipping {llm}: not pulled. Run 'ollama pull {llm}' to include it.")
    if not todo:
        print("\nNo candidate models are pulled. Nothing to benchmark.")
        return 1

    rows = []
    for compute_type in COMPUTE_TYPES:
        for llm in todo:
            print(f"\n=== distil-large-v3 {compute_type} + {llm}")
            try:
                row = bench_pair(audio, compute_type, llm)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                continue
            rows.append(row)
            print(f"  load {row['load_ms']}ms | asr {row['asr_ms']}ms | "
                  f"llm {row['llm_ms']}ms | total {row['total_ms']}ms")
            print(f"  peak VRAM {row['peak_mib']} MiB of 4096 | {row['residency']}")

    if not rows:
        return 1

    print("\n" + "=" * 96)
    print(f"{'compute_type':<14} {'llm':<14} {'asr':>7} {'llm':>7} {'total':>7} "
          f"{'peak':>9}  residency")
    print("-" * 96)
    for r in sorted(rows, key=lambda r: r["total_ms"]):
        fits = "" if r["peak_mib"] < 3900 else "  <-- OVER"
        print(f"{r['compute_type']:<14} {r['llm']:<14} {r['asr_ms']:>6}ms "
              f"{r['llm_ms']:>6}ms {r['total_ms']:>6}ms {r['peak_mib']:>6} MiB  "
              f"{r['residency']}{fits}")

    print("\nTranscript of the winning pair (check accuracy by eye, not just speed):")
    best = min(rows, key=lambda r: r["total_ms"])
    print(f"  {best['text']!r}")

    Path("bench_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nFull results in bench_results.json")
    print("Pick the fastest pair whose peak stays under ~3900 MiB and reads GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
