from __future__ import annotations

from datetime import datetime


def progress(message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)
