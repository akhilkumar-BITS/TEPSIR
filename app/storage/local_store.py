import asyncio
from pathlib import Path

from app.core.config import settings


class LocalObjectStore:
    """Filesystem-backed store rooted at ``LOCAL_STORAGE_DIR``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or settings.local_storage_dir).resolve()

    def _path(self, key: str) -> Path:
        # Resolve and verify containment to prevent path traversal via keys.
        target = (self.root / key).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Invalid storage key: {key}")
        return target

    async def ensure_ready(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            await asyncio.to_thread(path.unlink)
