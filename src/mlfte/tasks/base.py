"""Task interface: a task knows how to load data, build prompts, parse outputs and score."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from mlfte.config import Config


@dataclass(frozen=True)
class Example:
    """One supervised example, independent of tokenizer."""

    id: str
    question: str
    context: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Example:
        return cls(**{k: data[k] for k in ("id", "question", "context", "answer")})


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class Task(ABC):
    """Abstract task. Implementations are registered in ``mlfte.tasks``."""

    name: str = "base"

    @abstractmethod
    def load_examples(self, cfg: Config) -> list[Example]:
        """Load and normalise the raw dataset into ``Example`` objects."""

    @abstractmethod
    def system_prompt(self) -> str:
        """The system message used for training and inference."""

    @abstractmethod
    def user_prompt(self, example: Example) -> str:
        """The user message containing the schema and question."""

    def target(self, example: Example) -> str:
        """The completion the model should learn to produce."""
        return example.answer

    def messages(
        self, example: Example, few_shot: list[Example] | None = None
    ) -> list[ChatMessage]:
        """Chat messages for one example, optionally with fixed few-shot exemplars."""
        msgs: list[ChatMessage] = [ChatMessage("system", self.system_prompt())]
        for shot in few_shot or []:
            msgs.append(ChatMessage("user", self.user_prompt(shot)))
            msgs.append(ChatMessage("assistant", self.target(shot)))
        msgs.append(ChatMessage("user", self.user_prompt(example)))
        return msgs

    @abstractmethod
    def parse_output(self, text: str) -> str:
        """Extract the task answer from free-form model output."""

    def complexity(self, example: Example) -> dict[str, Any]:
        """Structural features of the gold answer, used for bucketed metrics."""
        return {}

    def bucket(self, example: Example) -> str:
        """Coarse difficulty bucket label for breakdowns."""
        return "all"

    def format_prompt_text(self, example: Example) -> str:
        """Human-readable prompt (used in data cards and error analysis)."""
        return f"{self.user_prompt(example)}\n\n-- answer --\n{self.target(example)}"
