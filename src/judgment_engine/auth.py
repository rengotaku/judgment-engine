"""Bearer 認証とプロセス内のレート制限。

鍵は起動時に 1 回だけ読む。設定が無ければ起動しない（fail-closed）。
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status


class BearerAuth:
    def __init__(self, api_key: str | None, allow_insecure: bool) -> None:
        self._key = api_key
        self._allow_insecure = allow_insecure

    @property
    def enabled(self) -> bool:
        return self._key is not None

    def verify(self, header: str | None) -> None:
        if not self.enabled:
            if self._allow_insecure:
                return
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="api key is not configured")
        if not header or not header.lower().startswith("bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        presented = header[7:].strip()
        if not hmac.compare_digest(presented, self._key or ""):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )


class RateLimiter:
    """1 分あたりの回数制限。鍵ごとに数える（鍵が無いときはクライアント IP）。"""

    def __init__(self, per_minute: int, window_sec: float = 60.0) -> None:
        self.per_minute = per_minute
        self.window = window_sec
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def identity(request: Request, header: str | None) -> str:
        if header and header.lower().startswith("bearer "):
            return "t:" + hashlib.sha256(header[7:].strip().encode("utf-8")).hexdigest()[:16]
        client = request.client.host if request.client else "unknown"
        return f"ip:{client}"

    def check(self, identity: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            bucket = self._events.setdefault(identity, deque())
            edge = now - self.window
            while bucket and bucket[0] < edge:
                bucket.popleft()
            if len(bucket) >= self.per_minute:
                return False, 0
            bucket.append(now)
            return True, self.per_minute - len(bucket)
