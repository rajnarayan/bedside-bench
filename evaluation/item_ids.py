"""Globally unique bedside rubric ``item_id`` values.

Each ID is ``md5(prompt)_{index}``: the question digest plus the item's
0-based position on that case. IDs are unique across the 500-case panel
because prompts do not repeat.
"""

from __future__ import annotations

import hashlib


def question_md5(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def item_id_for(prompt: str, index: int) -> str:
    return f"{question_md5(prompt)}_{index}"
