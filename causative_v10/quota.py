# -*- coding: utf-8 -*-
"""
quota.py — Quota tracking for causative corpus collection (v10)
================================================================
Tracks Pattern A / Pattern B acceptance counts against targets.
Thread-safe is NOT required (single-threaded pipeline).
"""
from collections import defaultdict
from typing import Dict

from causative_v10.config import CorpusConfig


class Quota:
    """Tracks how many Pattern A and Pattern B sentences have been accepted.

    Usage:
        quota = Quota(config)
        if quota.should_accept("A"):
            quota.accept("force", "A")
        if quota.done():
            # all targets met
    """

    def __init__(self, config: CorpusConfig) -> None:
        self.target_a: int = config.target_a
        self.target_b: int = config.target_b
        self.count_a: int = 0
        self.count_b: int = 0
        self.verb_counts_a: Dict[str, int] = defaultdict(int)
        self.verb_counts_b: Dict[str, int] = defaultdict(int)

    @property
    def total(self) -> int:
        """Total accepted sentences across both patterns."""
        return self.count_a + self.count_b

    def should_accept(self, pattern: str) -> bool:
        """Check if we still need sentences of this pattern type."""
        if pattern == "A":
            return self.count_a < self.target_a
        if pattern == "B":
            return self.count_b < self.target_b
        return False

    def accept(self, verb: str, pattern: str) -> None:
        """Record an accepted sentence."""
        if pattern == "A":
            self.count_a += 1
            self.verb_counts_a[verb] += 1
        else:
            self.count_b += 1
            self.verb_counts_b[verb] += 1

    def done(self) -> bool:
        """Check if both pattern targets have been reached."""
        return self.count_a >= self.target_a and self.count_b >= self.target_b

    def summary(self) -> str:
        """Human-readable progress string."""
        return (
            f"A:{self.count_a}/{self.target_a} "
            f"B:{self.count_b}/{self.target_b} "
            f"Total:{self.total}/{self.target_a + self.target_b}"
        )
