"""Input sanitizer — defends against prompt injection in job descriptions."""

import re
from typing import Tuple


class InputSanitizer:
    """Detects and neutralizes prompt injection attempts in user-provided text.

    Wraps sanitized input in XML delimiters before injecting into system prompts,
    making it structurally impossible for user content to escape its context.
    """

    # Patterns that indicate prompt injection attempts
    _INJECTION_PATTERNS = [
        r"ignore (previous|prior|all|above) instructions",
        r"disregard (your|the) (system|previous) (prompt|instructions)",
        r"you are now",
        r"act as (a |an )?(different|new|unrestricted)",
        r"<\s*system\s*>",
        r"<\s*instructions\s*>",
        r"\[INST\]",
        r"###\s*(instruction|system|prompt)",
        r"jailbreak",
    ]

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

    def sanitize(self, text: str) -> Tuple[str, bool]:
        """Clean user input and detect injection attempts.

        Returns (sanitized_text, was_suspicious).
        Suspicious inputs are still processed but flagged for audit logging.
        """
        suspicious = any(p.search(text) for p in self._COMPILED)

        # Strip null bytes and control characters (except newlines/tabs)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Wrap in XML delimiters to prevent injection into system prompts
        wrapped = f"<user_request>{cleaned}</user_request>"

        return wrapped, suspicious

    def wrap_for_prompt(self, text: str) -> str:
        """Wrap text in XML delimiters for safe injection into system prompts."""
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return f"<user_request>{cleaned}</user_request>"
