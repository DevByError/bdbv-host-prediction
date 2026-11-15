"""Stopwatch helpers for timing pipeline steps."""

import time
from contextlib import contextmanager
from typing import Optional, Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed = None
        return self

    def stop(self) -> float:
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def reset(self) -> "Timer":
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        return self

    def format(self) -> str:
        """Seconds under a minute, then m/s, then h/m."""
        if self.elapsed is None:
            return "Not measured"

        if self.elapsed < 60:
            return f"{self.elapsed:.2f}s"
        elif self.elapsed < 3600:
            minutes = int(self.elapsed // 60)
            seconds = self.elapsed % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(self.elapsed // 3600)
            minutes = int((self.elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        logger.info(f"Timer: {self.format()}")


@contextmanager
def timing(name: str = "Operation"):
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
        logger.info(f"{name} took: {timer.format()}")


class PerformanceTracker:
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def record(self, name: str, elapsed: float):
        """Adds to the running total for that name rather than replacing it."""
        if name not in self.times:
            self.times[name] = 0
            self.counts[name] = 0

        self.times[name] += elapsed
        self.counts[name] += 1

    def get_average(self, name: str) -> float:
        if name not in self.times or self.counts[name] == 0:
            return 0
        return self.times[name] / self.counts[name]

    def get_total(self, name: str) -> float:
        return self.times.get(name, 0)

    def get_summary(self) -> str:
        lines = ["Performance Summary:"]
        lines.append("-" * 40)

        for name in sorted(self.times.keys()):
            total = self.get_total(name)
            avg = self.get_average(name)
            count = self.counts[name]
            lines.append(f"{name}: total={total:.2f}s, avg={avg:.2f}s, count={count}")

        return "\n".join(lines)

    def log_summary(self):
        logger.info(self.get_summary())

    def reset(self):
        self.times.clear()
        self.counts.clear()
