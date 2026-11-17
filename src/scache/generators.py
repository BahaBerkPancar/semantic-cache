"""Answer generators for cache misses. `fake` is the default; `ollama` is a local model."""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol
from urllib.parse import urlparse

from .embeddings import OLLAMA_URL


class GeneratorError(RuntimeError):
    pass


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class FakeGenerator:
    """Deterministic, free. Counts calls so tests can prove a hit skipped the generator."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return f"fake answer to: {prompt}"


class OllamaGenerator:
    def __init__(self, model: str = "qwen2.5:3b", url: str = OLLAMA_URL) -> None:
        if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama URL must be a loopback address")
        self.model, self.url = model, url.rstrip("/")

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            self.url + "/api/generate", body, {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)["response"]
        except Exception as exc:  # network, timeout, bad JSON
            raise GeneratorError(str(exc)) from exc
