from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Tuple

import numpy as np

BatchKey = Hashable


@dataclass
class _InferenceRequest:
    """A single caller's request waiting to be batched."""

    texts: List[str]
    run_batch: Callable[[List[str]], Tuple[List[np.ndarray], int]]
    future: asyncio.Future[Tuple[List[np.ndarray], int]]
    text_count: int = field(init=False)

    def __post_init__(self):
        self.text_count = len(self.texts)


class InferenceBatcher:
    """Collects concurrent inference requests, groups compatible ones by batch
    key, and executes them as single batched calls to the underlying model.

    Uses a leader/follower pattern: the first caller to arrive when no
    inference is running becomes the leader and drives a drain-loop.
    Subsequent callers enqueue their request and ``await`` their
    :class:`asyncio.Future`.
    """

    def __init__(self) -> None:
        self._queue: Dict[BatchKey, List[_InferenceRequest]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._inference_running: bool = False

    def __deepcopy__(self, memo: dict) -> "InferenceBatcher":
        return self

    def __copy__(self) -> "InferenceBatcher":
        return self

    async def submit(
        self,
        batch_key: BatchKey,
        texts: List[str],
        run_batch: Callable[[List[str]], Tuple[List[np.ndarray], int]],
    ) -> Tuple[List[np.ndarray], int]:
        """Submit an inference request and await the result.

        Args:
            batch_key: Hashable key grouping compatible requests (all params
                except text).  Computed by the caller.
            texts: This caller's text(s), already normalised to a list.
            run_batch: A **synchronous** callable that accepts the combined
                text list and returns ``(wavs, sample_rate)``.  Will be run
                via :func:`asyncio.to_thread`.

        Returns:
            ``(wavs, sample_rate)`` -- only this caller's slice of the wavs.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Tuple[List[np.ndarray], int]] = loop.create_future()
        request = _InferenceRequest(texts=texts, run_batch=run_batch, future=future)

        is_leader = False
        async with self._lock:
            self._queue.setdefault(batch_key, []).append(request)
            if not self._inference_running:
                self._inference_running = True
                is_leader = True

        if is_leader:
            await self._run_batch_loop()

        return await future

    async def _run_batch_loop(self) -> None:
        """Leader loop: repeatedly drain the queue and run batched inference
        until no more requests are pending."""
        while True:
            async with self._lock:
                if not self._queue:
                    self._inference_running = False
                    return
                pending = self._queue
                self._queue = {}

            for _key, requests in pending.items():
                await self._execute_batch(requests)

    async def _execute_batch(self, requests: List[_InferenceRequest]) -> None:
        """Combine texts from *requests*, run inference once, distribute
        results back to each caller's future."""
        combined_texts: List[str] = []
        slices: List[Tuple[int, int]] = []
        for req in requests:
            start = len(combined_texts)
            combined_texts.extend(req.texts)
            slices.append((start, start + req.text_count))

        # All requests in the same batch share identical non-text params,
        # so any request's run_batch closure is equivalent.
        run_batch = requests[0].run_batch

        try:
            wavs, sr = await asyncio.to_thread(run_batch, combined_texts)

            for req, (start, end) in zip(requests, slices):
                if not req.future.done():
                    req.future.set_result((wavs[start:end], sr))
        except BaseException as exc:
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(exc)
