"""Persistent-cadence monitoring scheduler with bounded retries."""

from __future__ import annotations

import time
from threading import Event, Thread
from typing import Callable


class MonitorScheduler:
    def __init__(self, scan_all: Callable[[], None], interval_seconds: int, retry_seconds: int = 60) -> None:
        if interval_seconds < 1 or retry_seconds < 1:
            raise ValueError("scheduler intervals must be positive")
        self.scan_all = scan_all
        self.interval_seconds = interval_seconds
        self.retry_seconds = retry_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self.last_started_at: float | None = None
        self.last_completed_at: float | None = None
        self.last_error: str | None = None

    def run_once(self) -> None:
        self.last_started_at = time.time()
        try:
            self.scan_all()
            self.last_completed_at = time.time()
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            raise

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._loop, name="detection-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        delay = 0
        while not self._stop.wait(delay):
            try:
                self.run_once()
                delay = self.interval_seconds
            except Exception:
                delay = self.retry_seconds

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def status(self) -> dict:
        return {"running": bool(self._thread and self._thread.is_alive()), "lastStartedAt": self.last_started_at,
                "lastCompletedAt": self.last_completed_at, "lastError": self.last_error,
                "intervalSeconds": self.interval_seconds}
