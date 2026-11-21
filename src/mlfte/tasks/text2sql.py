"""Text-to-SQL task over ``b-mc2/sql-create-context``.

Prompt template (rendered through the tokenizer's chat template, see
``data/tokenize.py``)::

    system: You are a precise SQLite expert. Given a database schema and a question,
            write one SQL query that answers it. Respond with SQL only.

    user:   ### Schema:
            <CREATE TABLE ...;>

            ### Question:
            <natural language question>

            Write one SQLite query that answers the question. SQL only.

    assistant: <gold SQL>
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from mlfte.config import Config
from mlfte.tasks.base import Example, Task

SYSTEM_PROMPT = (
    "You are a precise SQLite expert. Given a database schema and a question, "
    "write one SQL query that answers it. Respond with SQL only, no explanation."
)

_PROMPT_TEMPLATE = (
    "### Schema:\n{context}\n\n"
    "### Question:\n{question}\n\n"
    "Write one SQLite query that answers the question. SQL only."
)

_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*", re.IGNORECASE)
_SQL_START_RE = re.compile(r"\b(select|with)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def stable_id(question: str, context: str) -> str:
    """Stable id from the (normalised) question+schema pair."""
    payload = f"{normalize_whitespace(question)}\x00{normalize_whitespace(context)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class Text2SQLTask(Task):
    name = "text2sql"

    # ----- loading -----
    def load_examples(self, cfg: Config) -> list[Example]:
        from datasets import load_dataset

        ds = load_dataset(cfg.data.dataset, split="train")
        examples: list[Example] = []
        for row in ds:
            question = normalize_whitespace(str(row["question"]))
            context = str(row["context"]).strip()
            answer = normalize_whitespace(str(row["answer"]))
            if not question or not context or not answer:
                continue
            examples.append(
                Example(
                    id=stable_id(question, context),
                    question=question,
                    context=context,
                    answer=answer,
                )
            )
        return examples

    # ----- prompting -----
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, example: Example) -> str:
        return _PROMPT_TEMPLATE.format(
            context=example.context.strip(), question=example.question.strip()
        )

    # ----- parsing -----
    def parse_output(self, text: str) -> str:
        """Extract the SQL statement from a raw generation.

        Handles markdown fences, chat-template leftovers and trailing prose. Returns an
        empty string when no SQL-looking content is found.
        """
        if not text:
            return ""
        cleaned = _FENCE_RE.sub("", text)
        cleaned = cleaned.replace("<|im_end|>", " ").replace("<|endoftext|>", " ")
        match = _SQL_START_RE.search(cleaned)
        if match:
            cleaned = cleaned[match.start() :]
        # Keep everything up to (and including) the first statement terminator.
        semi = cleaned.find(";")
        if semi != -1:
            cleaned = cleaned[: semi + 1]
        cleaned = cleaned.strip().strip("`").strip()
        return normalize_whitespace(cleaned)

    # ----- structure -----
    def complexity(self, example: Example) -> dict[str, Any]:
        features: dict[str, Any] = {
            "joins": 0,
            "aggregations": 0,
            "where": 0,
            "subqueries": 0,
            "group_by": 0,
            "order_by": 0,
            "limit": 0,
            "sql_chars": len(example.answer),
        }
        try:
            import sqlglot
            from sqlglot import exp

            parsed = sqlglot.parse_one(example.answer, dialect="sqlite")
        except Exception:
            return features
        features["joins"] = len(list(parsed.find_all(exp.Join)))
        features["aggregations"] = len(list(parsed.find_all(exp.AggFunc)))
        features["where"] = len(list(parsed.find_all(exp.Where)))
        features["subqueries"] = len(list(parsed.find_all(exp.Subquery)))
        features["group_by"] = len(list(parsed.find_all(exp.Group)))
        features["order_by"] = len(list(parsed.find_all(exp.Order)))
        features["limit"] = len(list(parsed.find_all(exp.Limit)))
        return features

    def bucket(self, example: Example) -> str:
        f = self.complexity(example)
        score = (
            f["joins"]
            + f["subqueries"]
            + (1 if f["aggregations"] else 0)
            + (1 if f["group_by"] else 0)
        )
        if score == 0:
            return "simple"
        if score <= 1:
            return "medium"
        return "complex"
