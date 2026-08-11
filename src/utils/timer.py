"""
Performance timing utilities.
"""

import time
from contextlib import contextmanager
from typing import Optional, Dict
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


class Timer:
    """
    Timer for performance measurement.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def reset(self) -> "Timer":
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        return self

    def format(self) -> str:
        """Format elapsed time as string."""
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
    """
    Context manager for timing code blocks.

    Args:
        name: Name of the operation

    Yields:
        Timer: Timer instance
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
        logger.info(f"{name} took: {timer.format()}")


class PerformanceTracker:
    """
    Track performance metrics across multiple operations.
    """

    def __init__(self):
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def record(self, name: str, elapsed: float):
        """Record a timing."""
        if name not in self.times:
            self.times[name] = 0
            self.counts[name] = 0

        self.times[name] += elapsed
        self.counts[name] += 1

    def get_average(self, name: str) -> float:
        """Get average time for an operation."""
        if name not in self.times or self.counts[name] == 0:
            return 0
        return self.times[name] / self.counts[name]

    def get_total(self, name: str) -> float:
        """Get total time for an operation."""
        return self.times.get(name, 0)

    def get_summary(self) -> str:
        """Get summary of all timings."""
        lines = ["Performance Summary:"]
        lines.append("-" * 40)

        for name in sorted(self.times.keys()):
            total = self.get_total(name)
            avg = self.get_average(name)
            count = self.counts[name]
            lines.append(f"{name}: total={total:.2f}s, avg={avg:.2f}s, count={count}")

        return "\n".join(lines)

    def log_summary(self):
        """Log the summary."""
        logger.info(self.get_summary())

    def reset(self):
        """Reset all metrics."""
        self.times.clear()
        self.counts.clear()
