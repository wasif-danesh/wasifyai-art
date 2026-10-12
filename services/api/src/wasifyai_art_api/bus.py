"""Job queue and per-job event log. Redis Streams in production; in memory for tests.

Events carry small JSON (progress, ids), never file bytes."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

from redis.asyncio import Redis
from redis.exceptions import ResponseError

JOBS_STREAM = "wsh:jobs"
GROUP = "workers"
TERMINAL = {"succeeded", "failed", "canceled"}


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    data: dict[str, Any]


class EventBus(Protocol):
    async def publish(self, job_id: str, type_: str, data: dict[str, Any]) -> None: ...

    async def read(self, job_id: str, after: str, block_ms: int) -> list[Event]: ...

    async def request_cancel(self, job_id: str) -> None: ...

    async def cancel_requested(self, job_id: str) -> bool: ...


class JobQueue(Protocol):
    async def enqueue(self, job_id: str) -> None: ...


class InMemoryBus:
    def __init__(self) -> None:
        self.events: dict[str, list[Event]] = {}
        self.queued: list[str] = []
        self.cancels: set[str] = set()

    async def request_cancel(self, job_id: str) -> None:
        self.cancels.add(job_id)

    async def cancel_requested(self, job_id: str) -> bool:
        return job_id in self.cancels

    async def enqueue(self, job_id: str) -> None:
        self.queued.append(job_id)

    async def publish(self, job_id: str, type_: str, data: dict[str, Any]) -> None:
        log = self.events.setdefault(job_id, [])
        log.append(Event(str(len(log) + 1), type_, data))

    async def read(self, job_id: str, after: str, block_ms: int) -> list[Event]:
        start = int(after) if after.isdigit() else 0
        for _ in range(max(1, block_ms // 50)):
            found = self.events.get(job_id, [])[start:]
            if found:
                return found
            await asyncio.sleep(0.05)
        return []


class RedisBus:
    def __init__(self, redis: Redis):
        self._r = redis

    @staticmethod
    def _key(job_id: str) -> str:
        return f"wsh:events:{job_id}"

    async def enqueue(self, job_id: str) -> None:
        await self._r.xadd(JOBS_STREAM, {"job_id": job_id})

    async def request_cancel(self, job_id: str) -> None:
        await self._r.set(f"wsh:cancel:{job_id}", "1", ex=3600)

    async def cancel_requested(self, job_id: str) -> bool:
        return bool(await self._r.exists(f"wsh:cancel:{job_id}"))

    async def publish(self, job_id: str, type_: str, data: dict[str, Any]) -> None:
        key = self._key(job_id)
        await self._r.xadd(
            key, {"type": type_, "data": json.dumps(data)}, maxlen=200, approximate=True
        )
        await self._r.expire(key, 3600)

    async def read(self, job_id: str, after: str, block_ms: int) -> list[Event]:
        res: Any = await self._r.xread({self._key(job_id): after or "0"}, block=block_ms, count=50)
        out: list[Event] = []
        for _stream, entries in res or []:
            for entry_id, fields in entries:
                f = {_s(k): _s(v) for k, v in fields.items()}
                out.append(Event(_s(entry_id), f["type"], json.loads(f["data"])))
        return out

    async def ensure_group(self) -> None:
        try:
            await self._r.xgroup_create(JOBS_STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
