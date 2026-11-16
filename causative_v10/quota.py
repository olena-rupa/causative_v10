# -*- coding: utf-8 -*-
"""Quota counter for Pattern A and Pattern B contexts."""

from collections import defaultdict
from typing import Dict

from causative_v10.config import CorpusConfig


class Quota:
    """Tracks accepted Pattern A and Pattern B contexts against target limits."""

    def __init__(self, config: CorpusConfig) -> None:
        self.target_a: int = config.target_a
        self.target_b: int = config.target_b
        self.count_a: int = 0
        self.count_b: int = 0
        self.verb_counts_a: Dict[str, int] = defaultdict(int)
        self.verb_counts_b: Dict[str, int] = defaultdict(int)

    @property
    def total(self) -> int:
        return self.count_a + self.count_b

    def should_accept(self, pattern: str) -> bool:
        if pattern == "A":
            return self.count_a < self.target_a
        if pattern == "B":
            return self.count_b < self.target_b
        return False

    def accept(self, verb: str, pattern: str) -> None:
        if pattern == "A":
            self.count_a += 1
            self.verb_counts_a[verb] += 1
        else:
            self.count_b += 1
            self.verb_counts_b[verb] += 1

    def done(self) -> bool:
        return self.count_a >= self.target_a and self.count_b >= self.target_b

    def summary(self) -> str:
        return (
            f"A:{self.count_a}/{self.target_a} "
            f"B:{self.count_b}/{self.target_b} "
            f"Total:{self.total}/{self.target_a + self.target_b}"
        )
