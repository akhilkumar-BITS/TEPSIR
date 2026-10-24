"""Object storage abstraction.

Documents are written once and read back when indexing or re-indexing, so the
interface is deliberately small. Local disk is the default for development;
MinIO provides the same contract for the Docker stack.
"""

from typing import Protocol


class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def delete(self, key: str) -> None: ...

    async def ensure_ready(self) -> None: ...


def build_object_store() -> ObjectStore:
    # Imported lazily so the unused backend's dependency is never touched.
    from app.core.config import settings

    if settings.storage_backend == "minio":
        from app.storage.minio_store import MinioObjectStore

        return MinioObjectStore()

    from app.storage.local_store import LocalObjectStore

    return LocalObjectStore()
