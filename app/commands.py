"""Spoken commands that become typed syntax: /skills, @mentions, #channels, line breaks.

Runs on the Whisper transcript before the LLM, so the model only ever sees finished
tokens, and the guard in llm.py can check it kept them. Pure text in, text out.

Everything is gated on the foreground app, because each phrase is ordinary speech
somewhere else: "new line" in a prompt to a coding agent, "tag" in an email.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Container, Optional

from . import winctx

# An @mention or #channel as typed. Not after a word character, so an email address and
# C# are left alone; a trailing full stop is punctuation, not part of the token.
TOKEN = re.compile(r"(?<!\w)([@#][A-Za-z](?:[\w/:-]|\.(?=\w))*)")

# What "tag everyone" means in each app. Apps not listed keep the spoken words.
EVERYONE = {"slack.exe": "channel", "discord.exe": "everyone",
            "teams.exe": "everyone", "ms-teams.exe": "everyone"}
CHANNEL_APPS = {"slack.exe", "discord.exe"}

_JOINERS = {"dot": ".", "slash": "/", "underscore": "_", "dash": "-", "hyphen": "-"}
_JOINER = r"\s+(dot|slash|underscore|dash|hyphen)\s+"
_WORD = r"\w+(?:[./-]\w+)*"  # Whisper sometimes writes "controller.py" itself
# One word, plus anything glued on by a spoken joiner: "app slash controller dot py".
# ponytail: one-word names, so "Rahul Sharma" types @Rahul and the app's popup picks the
# best match. Multi-word names need a spoken end marker; add one if that proves too loose.
_NAME = _WORD + r"(?:\s+(?:dot|slash|underscore|dash|hyphen)\s+" + _WORD + ")*"
_ALL = r"(?:everyone|everybody|all)\b"

_SLASH = re.compile(r"^\s*(?:slash\b|/)\s*(?P<rest>.+)$", re.I | re.S)
# A second command in the same dictation: "slash ponytail ultra slash caveman ultra".
_MORE_SLASH = re.compile(r"[\s,.]+slash\s+", re.I)
# Whisper hears "at the rate" as "add the rate" often enough to accept both.
_EVERYONE = re.compile(r"(?:\btag|\ba(?:t|dd) the rate|@)[\s,:]*" + _ALL, re.I)
# "tag" only before a capitalised word, which Whisper gives names: "tag Rahul" is a
# mention, "tag me later" and "the price tag is" are not.
_MENTION = re.compile(
    r"(?:\ba(?:t|dd) the rate\b(?!\s+of\b)"
    r"|\btag\b(?![\s,:]+" + _ALL + r")(?=[\s,:]+(?-i:[A-Z]))"
    r"|@)[\s,:]*(?P<name>" + _NAME + ")",
    re.I,
)
_HASHTAG = re.compile(r"\bhashtag[\s,:]+(?P<name>\w+(?:-\w+)*)", re.I)
# Whisper punctuates around the command ("First point. New line. Second"): swallow the
# commas and spaces around it, keep a full stop that ends the sentence before it.
_FORMATTING = [
    (re.compile(r"[\s,]*\bnew paragraph\b[\s,.:]*", re.I), "\n\n"),
    (re.compile(r"[\s,]*\bnew line\b[\s,.:]*", re.I), "\n"),
    (re.compile(r"[\s,]*\bbullet point\b[\s,.:]*", re.I), "\n- "),
]


@lru_cache(maxsize=1)
def installed_skills() -> frozenset[str]:
    """Claude Code skill and command names, lowercased, from the user's ~/.claude.

    ponytail: read once per run, so a newly installed skill needs an app restart, and a
    project's own .claude/skills is not scanned because the workspace path is unknown.
    """
    root = Path.home() / ".claude"
    skills = [*root.glob("skills/*/SKILL.md"),
              *root.glob("plugins/cache/*/*/*/skills/*/SKILL.md")]
    commands = [*root.glob("commands/*.md"),
                *root.glob("plugins/cache/*/*/*/commands/*.md")]
    return frozenset(n.lower() for n in
                     [p.parent.name for p in skills] + [p.stem for p in commands])


def slash_command(rest: str, skills: Container[str]) -> str:
    """'ponytail review' -> '/ponytail-review' if that skill exists, else
    '/ponytail review'. Longest match wins, so skill names beat arguments."""
    words = rest.strip().rstrip(".!?").split()
    if not words:
        return "/"
    norm = [w.strip(",.!?").lower() for w in words]
    for k in range(min(4, len(words)), 0, -1):
        # Joined with hyphens, or with nothing for a name Whisper split: "pony tail".
        for name in ("-".join(norm[:k]), "".join(norm[:k])):
            if name in skills:
                return " ".join(["/" + name, *words[k:]])
    return " ".join(["/" + norm[0], *words[1:]])


def _join(name: str) -> str:
    return re.sub(_JOINER, lambda m: _JOINERS[m[1].lower()], name, flags=re.I)


def rewrite(text: str, fg: Optional[winctx.Foreground], mention_apps: Container[str],
            skills: Optional[Container[str]] = None) -> str:
    """Turn spoken commands into what the foreground app expects typed.

    `mention_apps` is the set of exe names whose mention popup can be driven; mentions
    are left as spoken everywhere else.
    """
    if fg is None:
        return text
    exe = (fg.exe or "").lower()

    if fg.context in (winctx.CODE, winctx.TERMINAL):
        # Claude Code's panel and CLI. Only at the start, where a command has to be.
        slash = _SLASH.match(text)
        if slash:
            skills = installed_skills() if skills is None else skills
            first, *more = _MORE_SLASH.split(slash["rest"])
            out = [slash_command(first, skills)]
            for part in more:
                # Only a known skill: "app slash controller" is a path, not a command.
                command = slash_command(part, skills)
                out.append(command if command.split()[0][1:] in skills else "slash " + part)
            return " ".join(out)
    else:
        # Not in code, where "add a new line" is ordinary speech, and never in a
        # terminal, where a pasted line break runs the command.
        for pattern, replacement in _FORMATTING:
            text = pattern.sub(replacement, text)

    if exe in mention_apps:
        if exe in EVERYONE:
            text = _EVERYONE.sub("@" + EVERYONE[exe], text)
        text = _MENTION.sub(lambda m: "@" + _join(m["name"]), text)
        if exe in CHANNEL_APPS:
            text = _HASHTAG.sub(lambda m: "#" + m["name"], text)
    return text.strip()
