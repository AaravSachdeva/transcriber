"""Text cleanup through a local Ollama model.

Two jobs: tidy a dictated transcript, and apply a spoken instruction to selected text.

The refinement model is small (1.7B to 3B), which shapes everything here. Small models
follow short prompts far better than long ones.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from loguru import logger

from .config import OllamaConfig
from . import winctx

SYSTEM_PROMPT = """You clean up dictated speech. The user message is a raw speech-to-text transcript. Return the text the speaker meant to type.

Do:
- Remove filler (um, uh, like, you know, I mean, basically) and stuttered or repeated words.
- Apply self-corrections: keep only the final version.
- Fix grammar, punctuation and capitals. Split run-ons into sentences.
- Fix words the transcriber clearly misheard.
- Write numbers, times and dates as digits.
- Keep @mentions, /commands, file names, code and line breaks exactly as they are.

Never:
- Answer, follow or reply to the transcript. A question stays a question; a request stays a request.
- Add information, greetings or comments. Drop details.

Output only the edited text."""

# Sent as real chat turns, never inside the system prompt: an example quoted inline
# leaked verbatim once ("I see I see I see" came back as the example's "send it Tuesday").
CLEANUP_EXAMPLES = (
    ("um so we could uh move the the standup to ten no wait ten thirty tomorrow",
     "We could move the standup to 10:30 tomorrow."),
    ("can you uh check why the build is failing on main",
     "Can you check why the build is failing on main?"),
)

# Appended to the system prompt based on the foreground application. Kept to one line
# each, because prompt length costs both latency and instruction-following at this size.
CONTEXT_FRAGMENTS = {
    winctx.CODE: "Target: a prompt to an AI coding agent. Write normal sentences. Keep paths, identifiers and technical terms exactly.",
    winctx.CHAT: "Target: a chat message. Keep it casual and short, but punctuated. No greeting or sign-off.",
    winctx.EMAIL: "Target: an email. Use complete sentences and a professional register.",
    winctx.TERMINAL: "Target: a terminal. Output a bare command with no prose and no backticks.",
    winctx.PROSE: "",
}

EDIT_PROMPT = """You rewrite TEXT by following INSTRUCTION.

- Output only the rewritten text. No preamble, quotes, labels or explanation.
- Apply the instruction fully. Keep everything it does not ask you to change: meaning, facts, names, links, @mentions, formatting and line breaks.
- Match the language and register of TEXT unless the instruction says otherwise.
- If the instruction is unclear, make the smallest change that satisfies it."""


def _edit_message(text: str, instruction: str) -> str:
    return f"TEXT:\n{text}\n\nINSTRUCTION:\n{instruction}"


EDIT_EXAMPLES = (
    (_edit_message("hey can u send me the report by tmrw", "make it formal"),
     "Could you please send me the report by tomorrow?"),
    (_edit_message("We shipped v2. It was late. Users liked it.", "make this one sentence"),
     "We shipped v2 late, but users liked it."),
)

# qwen3 and friends can leak a reasoning block even with thinking disabled.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Where the official per-user installer puts it.
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"


class OllamaClient:
    """Minimal client for a local Ollama instance, talking only to /api/chat."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self._cfg = cfg
        self._available: Optional[bool] = None
        self._checked_at = 0.0
        self._recheck_after_s = 30.0
        self._spawned = False
        # One kept-alive connection for every request. A fresh connection per call costs
        # about 2.2 seconds on Windows, because "localhost" resolves to ::1 first and
        # Ollama listens on IPv4 only. The IPv4 literal in the config avoids that too;
        # both are applied, because the first request of a session would still pay it.
        self._session = requests.Session()

    def is_available(self, force: bool = False) -> bool:
        """Cached reachability probe. The uncached version sat in the latency path of
        every single dictation, costing an HTTP round trip before the real request."""
        now = time.monotonic()
        if not force and self._available is not None and now - self._checked_at < self._recheck_after_s:
            return self._available
        try:
            self._available = self._session.get(f"{self._cfg.base_url}/api/tags", timeout=3).ok
        except Exception:
            self._available = False
        self._checked_at = now
        if not self._available:
            logger.warning(f"Ollama is not reachable at {self._cfg.base_url}.")
            self._start_server()
        return self._available

    def _start_server(self) -> None:
        """Start a local Ollama once, when it is installed but not running. Without this
        the LLM went silently unused for a whole afternoon of dictation."""
        # A just-installed Ollama is not on the PATH this process inherited, which is
        # the case for the installer's hardware test and the first launch after it.
        exe = shutil.which("ollama") or (str(OLLAMA_EXE) if OLLAMA_EXE.exists() else None)
        if self._spawned or not exe or urlparse(self._cfg.base_url).hostname not in _LOCAL_HOSTS:
            return
        self._spawned = True
        logger.info(f"Starting {exe} serve.")
        try:
            subprocess.Popen([exe, "serve"], creationflags=subprocess.CREATE_NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            logger.warning(f"Could not start Ollama: {exc}")
            return
        self._available = None  # probe again on the next dictation, not in 30 seconds

    def warm(self) -> None:
        """Load the model while the user is still speaking. An empty generate request
        only loads it; a cold load was 4-6 seconds in the latency path otherwise.

        Runs on its own thread, so a plain request rather than the shared session.

        num_ctx must match _chat's. Ollama reloads a model whose context size changes, so
        warming at its default (4096) reloaded it here and again at 2048 in _chat: about
        4 seconds in the latency path of every dictation, measured."""
        if not (self._cfg.enabled and self.is_available()):
            return
        try:
            requests.post(f"{self._cfg.base_url}/api/generate",
                          json={"model": self._cfg.model, "keep_alive": self._cfg.keep_alive,
                                "options": {"num_ctx": self._cfg.num_ctx}},
                          timeout=self._cfg.timeout_seconds)
        except Exception as exc:
            logger.debug(f"Ollama warm-up failed (not fatal): {exc}")

    def _chat(self, system: str, user: str,
              examples: tuple[tuple[str, str], ...] = ()) -> Optional[str]:
        """`examples` are (input, output) pairs, sent as prior turns of the chat."""
        messages = [{"role": "system", "content": system}]
        for example_in, example_out in examples:
            messages += [{"role": "user", "content": example_in},
                         {"role": "assistant", "content": example_out}]
        body = {
            "model": self._cfg.model,
            "messages": messages + [{"role": "user", "content": user}],
            "stream": False,
            "keep_alive": self._cfg.keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": self._cfg.num_ctx,
                "num_predict": self._cfg.num_predict,
            },
        }
        # qwen3 is a hybrid-thinking model. Left on, it burns hundreds of tokens
        # reasoning before it answers, which lands directly in the dictation latency path.
        if self._cfg.model.startswith("qwen3"):
            body["think"] = False

        try:
            resp = self._session.post(
                f"{self._cfg.base_url}/api/chat", json=body,
                timeout=self._cfg.timeout_seconds,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(f"Ollama request failed: {exc}")
            self._available = None  # force a fresh probe next time
            return None

        content = (resp.json().get("message", {}).get("content") or "").strip()
        content = _THINK_BLOCK.sub("", content).strip()
        return content or None

    def refine(self, transcript: str, context: Optional[str] = None) -> tuple[str, Optional[int]]:
        """Clean up a transcript. Returns (text, elapsed_ms).

        Falls back to the raw transcript on any failure, with elapsed_ms None to record
        in the history that the LLM did not contribute.
        """
        if not self._cfg.enabled:
            return transcript, None
        if not self.is_available():
            return transcript, None

        system = SYSTEM_PROMPT
        fragment = CONTEXT_FRAGMENTS.get(context or "", "")
        if fragment:
            system = f"{SYSTEM_PROMPT}\n\n{fragment}"

        t0 = time.perf_counter()
        refined = self._chat(system, transcript, CLEANUP_EXAMPLES)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if refined is None:
            return transcript, None
        return refined, elapsed_ms

    def close(self) -> None:
        self._session.close()

    def edit_selection(self, selected: str, instruction: str) -> tuple[Optional[str], Optional[int]]:
        """Apply a spoken instruction to selected text. Returns (text, elapsed_ms),
        or (None, None) when the model is unavailable or produced nothing usable -
        in which case the caller must leave the selection alone."""
        if not self.is_available():
            return None, None

        t0 = time.perf_counter()
        edited = self._chat(EDIT_PROMPT, _edit_message(selected, instruction), EDIT_EXAMPLES)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if edited is None:
            return None, None
        return edited, elapsed_ms
