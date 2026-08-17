from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SingleInstanceError(RuntimeError):
    pass


@contextmanager
def single_instance_lock(path: str | Path) -> Iterator[None]:
    """Hold an OS file lock so only one runner can use a demo state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SingleInstanceError("Another v5.1 demo runner is already using this state") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise SingleInstanceError("Another v5.1 demo runner is already using this state") from exc
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()
