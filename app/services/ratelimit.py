"""In-memory per-visitor rate limiting / abuse guardrail.

Counters live in process memory and reset daily. For local mode this just bounds
how hard a single visitor can hit the (CPU-bound) local model; in Bedrock mode
(Part 2) the same limiter is the app-level ceiling that keeps spend predictable.
The real hard cost ceiling in Bedrock mode is the AWS budget/quota itself.
"""

import threading
from datetime import date

from app.config import settings

_lock = threading.Lock()
_queries: dict[str, int] = {}
_uploads: dict[str, int] = {}
_global_count = 0
_current_day = ""


def _roll_day() -> None:
    global _current_day, _global_count
    today = date.today().isoformat()
    if today != _current_day:
        _current_day = today
        _global_count = 0
        _queries.clear()
        _uploads.clear()


def _consume(bucket: dict[str, int], ip: str, per_ip_limit: int) -> tuple[bool, int]:
    global _global_count
    with _lock:
        _roll_day()
        if _global_count >= settings.global_daily_cap:
            return False, 0
        used = bucket.get(ip, 0)
        if used >= per_ip_limit:
            return False, 0
        used += 1
        bucket[ip] = used
        _global_count += 1
        return True, max(0, per_ip_limit - used)


def _remaining(bucket: dict[str, int], ip: str, per_ip_limit: int) -> int:
    with _lock:
        _roll_day()
        if _global_count >= settings.global_daily_cap:
            return 0
        return max(0, per_ip_limit - bucket.get(ip, 0))


def consume_quota(ip: str) -> tuple[bool, int]:
    return _consume(_queries, ip, settings.free_queries_per_day)


def remaining_quota(ip: str) -> int:
    return _remaining(_queries, ip, settings.free_queries_per_day)


def consume_upload(ip: str) -> tuple[bool, int]:
    return _consume(_uploads, ip, settings.free_uploads_per_day)


def remaining_uploads(ip: str) -> int:
    return _remaining(_uploads, ip, settings.free_uploads_per_day)
