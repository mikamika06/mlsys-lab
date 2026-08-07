import asyncio
import inspect
from typing import Any, Callable, List, Tuple

class BatchQueue:
    def __init__(self, fn: Callable, max_batch_size: int, batch_wait_timeout_s: float):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError("The @serve.batch decorator can only be applied to async functions.")
        self.fn = fn
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.queue: List[Tuple[Any, asyncio.Future]] = []
        self.timer_task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.batch_count = 0

    async def _process_batch(self):
        async with self.lock:
            if not self.queue:
                return
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
            self.batch_count += 1

        args = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        try:
            results = await self.fn(args)
            for fut, res in zip(futures, results):
                if not fut.done():
                    fut.set_result(res)
        except Exception as exc:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(exc)

        async with self.lock:
            if self.queue and self.timer_task is None:
                self.timer_task = asyncio.create_task(self._wait_and_flush())

    async def _wait_and_flush(self):
        await asyncio.sleep(self.batch_wait_timeout_s)
        async with self.lock:
            self.timer_task = None
        await self._process_batch()

    async def __call__(self, arg: Any) -> Any:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        async with self.lock:
            self.queue.append((arg, fut))
            trigger_flush = len(self.queue) >= self.max_batch_size
            if not trigger_flush and self.timer_task is None:
                self.timer_task = asyncio.create_task(self._wait_and_flush())

        if trigger_flush:
            if self.timer_task:
                self.timer_task.cancel()
                self.timer_task = None
            await self._process_batch()

        return await fut

def serve_batch(max_batch_size: int = 10, batch_wait_timeout_s: float = 0.05):
    def decorator(fn: Callable):
        queue = BatchQueue(fn, max_batch_size, batch_wait_timeout_s)
        async def wrapper(arg: Any) -> Any:
            return await queue(arg)
        wrapper.__batch_queue__ = queue
        return wrapper
    return decorator
