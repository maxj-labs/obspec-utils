"""Reads pinned to one object revision with If-Match."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Protocol

from obspec import Get, GetOptions, GetResult, Head, ObjectMeta


class _GetHead(Get, Head, Protocol): ...


def head_pinnable(store: Head, path: str) -> tuple[int, str]:
    """HEAD the object and return its size and strong ETag, which pinning needs."""
    meta = store.head(path)
    e_tag = meta.get("e_tag")
    if not e_tag:
        raise ValueError(
            f"pin_etag=True, but the store reports no ETag for {path!r} to pin reads to"
        )
    if e_tag.startswith("W/"):
        raise ValueError(
            f"pin_etag=True, but {path!r} has the weak ETag {e_tag!r}; If-Match "
            "needs a strong one"
        )
    return meta["size"], e_tag


class PinnedStore:
    """A view of a store in which every read of one object carries If-Match.

    [`get()`][obspec.Get] adds `if_match`; `get_range()` and `get_ranges()` go
    through `get()` with a `range`, since the obspec range methods take no
    options; `head()` is passed through. A read of a modified object raises the
    store's precondition error (`obstore.exceptions.PreconditionError` for
    obstore stores, `obspec.exceptions.PreconditionError` for obspec-utils' own
    stores). Ranges of one `get_ranges()` call run on a thread pool; the first
    failure cancels the ranges not yet started.
    """

    def __init__(
        self, store: _GetHead, path: str, e_tag: str, max_workers: int = 16
    ) -> None:
        self._store = store
        self._path = path
        self._e_tag = e_tag
        self._max_workers = max_workers

    def _check_path(self, path: str) -> None:
        if path != self._path:
            raise ValueError(f"store is pinned to {self._path!r}, not {path!r}")

    def head(self, path: str) -> ObjectMeta:
        return self._store.head(path)

    def get(self, path: str, *, options: GetOptions | None = None) -> GetResult:
        self._check_path(path)
        pinned: GetOptions = {**(options or {}), "if_match": self._e_tag}
        return self._store.get(path, options=pinned)

    def get_range(
        self,
        path: str,
        *,
        start: int,
        end: int | None = None,
        length: int | None = None,
    ) -> bytes:
        if end is None:
            if length is None:
                raise ValueError("Must provide end or length")
            end = start + length
        return bytes(self.get(path, options={"range": (start, end)}).buffer())

    def get_ranges(
        self,
        path: str,
        *,
        starts: Sequence[int],
        ends: Sequence[int] | None = None,
        lengths: Sequence[int] | None = None,
    ) -> list[bytes]:
        if ends is None:
            if lengths is None:
                raise ValueError("Must provide ends or lengths")
            ends = [start + length for start, length in zip(starts, lengths)]
        if len(starts) == 1:
            return [self.get_range(path, start=starts[0], end=ends[0])]
        with ThreadPoolExecutor(min(self._max_workers, len(starts))) as pool:
            futures = [
                pool.submit(self.get_range, path, start=start, end=end)
                for start, end in zip(starts, ends)
            ]
            done, _ = wait(futures, return_when=FIRST_EXCEPTION)
            for future in done:
                if future.exception() is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                    future.result()  # raises
            return [future.result() for future in futures]
