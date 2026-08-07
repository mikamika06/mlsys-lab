import asyncio
import inspect
from typing import Any, Callable, List

class ServeBatchTypeError(TypeError):
    """Raised when @serve.batch decorates a non-async function."""
    pass

class BatchExecutor:
    def __init__(self, fn: Callable, max_batch_size: int, batch_wait_timeout_s: float):
        if not inspect.iscoroutinefunction(fn):
            raise ServeBatchTypeError(
                f"The @serve.batch decorator can only be applied to async functions, got {fn}"
            )
        self.fn = fn
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.queue: List[Tuple[Any, asyncio.Future]] = []
        self.timer_task = None
        self.lock = asyncio.Lock()

    async def _flush(self):
        async with self.lock:
            if not self.queue:
                return
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]

        args_list = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        try:
            results = await self.fn(args_list)
            for fut, res in zip(futures, results):
                if not fut.done():
                    fut.set_result(res)
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

        async with self.lock:
            if self.queue and self.timer_task is None:
                self.timer_task = asyncio.create_task(self._wait_and_flush())

    async def _wait_and_flush(self):
        await asyncio.sleep(self.batch_wait_timeout_s)
        async with self.lock:
            self.timer_task = None
        await self._flush()

    async def __call__(self, arg: Any) -> Any:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        async with self.lock:
            self.queue.append((arg, fut))
            should_flush_now = len(self.queue) >= self.max_batch_size
            if not should_flush_now and self.timer_task is None:
                self.timer_task = asyncio.create_task(self._wait_and_flush())

        if should_flush_now:
            if self.timer_task:
                self.timer_task.cancel()
                self.timer_task = None
            await self._flush()

        return await fut

def serve_batch(max_batch_size: int = 10, batch_wait_timeout_s: float = 0.05):
    def decorator(fn: Callable):
        executor = BatchExecutor(fn, max_batch_size, batch_wait_timeout_s)
        async def wrapper(arg: Any) -> Any:
            return await executor(arg)
        wrapper.__serve_batch_executor__ = executor
        return wrapper
    return decorator
