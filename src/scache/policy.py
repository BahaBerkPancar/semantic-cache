"""Never-cache policy: which prompts may not be served from, or stored in, the cache."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .guards import is_personal, time_scope

# Time scopes (see guards.time_scope) that make an answer stale by itself.
VOLATILE_TIME = frozenset({"today", "yesterday", "tomorrow", "now"})
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_ERROR_ANSWER = re.compile(
    r"^\s*(error\b|\[error\]|traceback \(most recent call last\))", re.IGNORECASE
)


@dataclass(frozen=True)
class NeverCachePolicy:
    """Each rule can be switched off. `check_*` return a reason string, or None when cacheable."""

    personal_without_user: bool = True
    volatile_time: bool = True
    explicit_dates: bool = True
    error_answers: bool = True
    volatile_words: frozenset[str] = field(default=VOLATILE_TIME)

    def check_prompt(self, prompt: str, user_id: str | None, no_cache: bool = False) -> str | None:
        if no_cache:
            return "no_cache requested"
        if self.personal_without_user and user_id is None and is_personal(prompt):
            return "personal prompt without user id"
        if self.volatile_time and (time_scope(prompt) & self.volatile_words):
            return "time-sensitive prompt"
        if self.explicit_dates and _ISO_DATE.search(prompt):
            return "prompt contains a date"
        return None

    def check_answer(self, answer: str, is_error: bool = False) -> str | None:
        if self.error_answers and (is_error or _ERROR_ANSWER.search(answer)):
            return "answer flagged as error"
        return None
