"""Pluggable tasks. Add a module implementing ``Task`` and register it here."""

from __future__ import annotations

from mlfte.tasks.base import Example, Task
from mlfte.tasks.text2sql import Text2SQLTask

TASKS: dict[str, type[Task]] = {"text2sql": Text2SQLTask}


def get_task(name: str = "text2sql") -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]()


__all__ = ["Example", "Task", "Text2SQLTask", "TASKS", "get_task"]
