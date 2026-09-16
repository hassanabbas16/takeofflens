"""Storage abstraction.

Local disk today. The interface is deliberately narrow (put/get/path/exists) so an S3
backend can be dropped in without touching the pipeline - see README "how to scale".
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        """Write bytes under ``key``; returns the stored key."""

    @abstractmethod
    def put_file(self, key: str, source: Path) -> str:
        """Copy a local file under ``key``; returns the stored key."""

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def local_path(self, key: str) -> Path:
        """Filesystem path for ``key``.

        S3 implementations will need to materialise to a temp file here; OpenCV and
        PyMuPDF both want a real path.
        """


class LocalStorage(Storage):
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Guard against a key escaping the storage root via traversal.
        path = (self.root / key).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes storage root: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def put_file(self, key: str, source: Path) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        return key

    def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def local_path(self, key: str) -> Path:
        return self._resolve(key)
