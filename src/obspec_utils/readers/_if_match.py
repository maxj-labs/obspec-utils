"""Range reads pinned to one object revision with If-Match."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from obspec import Get, Head


def require_e_tag(store: Head, path: str) -> tuple[int, str]:
    """HEAD the object and return its size and ETag, which pinned reads need."""
    meta = store.head(path)
    e_tag = meta.get("e_tag")
    if not e_tag:
        raise ValueError(
            f"pin_etag=True, but the store reports no ETag for {path!r} "
            "to pin reads to"
        )
    return meta["size"], e_tag


def get_ranges_if_match(
    store: Get,
    path: str,
    *,
    starts: Sequence[int],
    lengths: Sequence[int],
    e_tag: str,
    max_workers: int = 16,
) -> list[bytes]:
    """Fetch byte ranges, each sent with If-Match so a changed object fails.

    [`get_ranges()`][obspec.GetRanges] takes no options, so each range is a
    [`get()`][obspec.Get] with a `range` and `if_match`; several run on a
    thread pool. Raises the store's precondition error (e.g.
    `obstore.exceptions.PreconditionError`) if the object's ETag no longer
    matches.
    """

    def fetch(start: int, length: int) -> bytes:
        options = {"range": (start, start + length), "if_match": e_tag}
        return bytes(store.get(path, options=options).buffer())  # type: ignore[arg-type]

    if len(starts) == 1:
        return [fetch(starts[0], lengths[0])]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(starts))) as pool:
        return list(pool.map(fetch, starts, lengths))
