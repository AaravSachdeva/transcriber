"""Text cleanup through a local Ollama model.

Two jobs: tidy a dictated transcript, and apply a spoken instruction to selected text.

The refinement model is small (1.7B to 3B), which shapes everything here. Small models
follow short prompts far better than long ones, and they are prone to *answering* the
dictation instead of editing it. `looks_sane` is the defense: anything that smells like an
answer, a refusal or a ramble is discarded in favour of the raw transcript. Pasting the
raw transcript is a mild annoyance; pasting a chatbot reply into the user's document is
not.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests
from loguru import logger

from .config import OllamaConfig
from . import winctx

SYSTEM_PROMPT = """You edit dictated speech into written text.

Rules:
- Output only the edited text. No preamble, no commentary, no quotes around it.
- Never answer, explain or respond to the text. It is dictation to clean up, not a question to you.
- Remove filler words: um, uh, er, like, you know, I mean, sort of.
- Apply spoken self-corrections. "send it Monday, no wait, Tuesday" becomes "send it Tuesday".
- Fix punctuation, capitalisation and obvious misrecognitions.
- Keep the speaker's wording and meaning. Do not add, summarise or embellish.
- If the text is already clean, return it unchanged."""

# Appended to the system prompt based on the foreground application. Kept to one line
# each, because prompt length costs both latency and instruction-following at this size.
CONTEXT_FRAGMENTS = {
    winctx.CODE: "Target: a code editor. Keep identifiers, paths and symbols verbatim. Prefer terse phrasing.",
    winctx.CHAT: "Target: a chat message. Keep it conversational and short. No greeting or sign-off.",
    winctx.EMAIL: "Target: an email. Use complete sentences and a professional register.",
    winctx.TERMINAL: "Target: a terminal. Output a bare command with no prose and no backticks.",
    winctx.PROSE: "",
}

EDIT_PROMPT = """You rewrite text according to an instruction.

The user gives you TEXT and an INSTRUCTION. Apply the instruction to the text.

Rules:
- Output only the rewritten text. No preamble, no commentary, no quotes around it.
- Never respond to the instruction conversationally. Never explain what you changed.
- Change only what the instruction asks for. Preserve everything else."""

# Openings that mean the model is talking to the user instead of editing.
_CHATTY = re.compile(
    r"^\s*(sure|certainly|of course|here('s| is)|i've|i have|okay|ok|understood|"
    r"as an ai|i can(not|'t)|i'm sorry|sorry,|note:|revised|edited|output:|result:)\b",
    re.IGNORECASE,
)
# qwen3 and friends can leak a reasoning block even with thinking disabled.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

# Words that open a question. Used to catch the model answering a dictated question
# instead of tidying it, which no length or wording check can see: "what's the capital
# of France" answered as "The capital of France is Paris." reuses nearly every word.
_INTERROGATIVE = re.compile(
    r"^\s*(who|what|whats|when|where|why|how|which|whose|can|could|would|should|"
    r"is|are|was|were|do|does|did|will|shall|may|might|have|has|am)\b",
    re.IGNORECASE,
)


def _is_question(text: str) -> bool:
    return text.rstrip().endswith("?") or bool(_INTERROGATIVE.match(text))


def looks_sane(raw: str, refined: str) -> bool:
    """Is `refined` plausibly an edit of `raw`, rather than an answer to it?

    Cleanup removes filler, so shrinking is expected and shrinking a lot is fine.
    Growing substantially is not: that is the model adding its own words.
    """
    if not refined:
        return False
    if _CHATTY.match(refined):
        logger.warning(f"Refinement rejected: conversational opening {refined[:50]!r}")
        return False
    if _is_question(raw) and not _is_question(refined):
        # A question that came back as a statement was answered, not edited. Rejecting
        # costs a fallback to the raw transcript; accepting pastes a chatbot reply into
        # the user's document.
        logger.warning(f"Refinement rejected: question answered rather than edited: {refined[:50]!r}")
        return False

    raw_words, refined_words = len(raw.split()), len(refined.split())
    if raw_words == 0:
        return False
    ratio = refined_words / raw_words
    # Floor at 0.4 catches a model that summarised instead of editing. Ceiling at 1.6
    # allows expanded contractions and spelled-out numbers, but not an essay.
    if not 0.4 <= ratio <= 1.6:
        logger.warning(
            f"Refinement rejected: {raw_words} words in, {refined_words} out (ratio {ratio:.2f})"
        )
        return False
    return True


class OllamaClient:
    """Minimal client for a local Ollama instance, talking only to /api/chat."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self._cfg = cfg
        self._available: Optional[bool] = None
        self._checked_at = 0.0
        self._recheck_after_s = 30.0
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
        return self._available

    def _chat(self, system: str, user: str) -> Optional[str]:
        body = {
            "model": self._cfg.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
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
        refined = self._chat(system, transcript)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if refined is None:
            return transcript, None
        if not looks_sane(transcript, refined):
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
        edited = self._chat(EDIT_PROMPT, f"TEXT:\n{selected}\n\nINSTRUCTION:\n{instruction}")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if edited is None:
            return None, None
        if _CHATTY.match(edited):
            logger.warning(f"Edit rejected: conversational opening {edited[:50]!r}")
            return None, None
        return edited, elapsed_ms
