"""Small console-formatting helpers for readable CLI output."""
from __future__ import annotations

import os
import sys


_RESET = "\033[0m"
_STYLES = {
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "bold": "\033[1m",
}


def supports_color() -> bool:
    """Return whether ANSI color should be emitted."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = getattr(sys.stdout, "_console", sys.stdout)
    return bool(getattr(stream, "isatty", lambda: False)())


def colorize(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles when color output is enabled."""
    if not supports_color() or not styles:
        return text
    prefix = "".join(_STYLES[name] for name in styles if name in _STYLES)
    return f"{prefix}{text}{_RESET}" if prefix else text


def status_text(label: str, kind: str) -> str:
    """Return a consistently styled status label."""
    style_map = {
        "info": ("cyan", "bold"),
        "ok": ("green", "bold"),
        "warn": ("yellow", "bold"),
        "error": ("red", "bold"),
        "stage": ("blue", "bold"),
    }
    return colorize(label, *style_map.get(kind, ("bold",)))
