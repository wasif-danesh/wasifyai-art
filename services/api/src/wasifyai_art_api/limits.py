"""Per-user rate limits: a fixed window counter in Redis (in memory for tests)."""

import time
from typing import Protocol

from redis.asyncio import Redis


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_s: int) -> bool:
        """Count one use; False when the limit for the current window is already reached."""
        ...


class InMemoryLimiter:
    def __init__(self) -> None:
        self._windows: dict[str, tuple[float, int]] = {}

    async def hit(self, key: str, limit: int, window_s: int) -> bool:
        now = time.monotonic()
        start, count = self._windows.get(key, (now, 0))
        if now - start >= window_s:
            start, count = now, 0
        if count >= limit:
            return False
        self._windows[key] = (start, count + 1)
        return True


class RedisLimiter:
    def __init__(self, redis: Redis):
        self._r = redis

    async def hit(self, key: str, limit: int, window_s: int) -> bool:
        k = f"wsh:rl:{key}"
        n = await self._r.incr(k)
        if n == 1:
            await self._r.expire(k, window_s)
        return int(n) <= limit
