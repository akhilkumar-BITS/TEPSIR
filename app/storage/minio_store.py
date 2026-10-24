import asyncio
import io
from functools import partial

from minio import Minio

from app.core.config import settings


class MinioObjectStore:
    """S3-compatible store; the MinIO client is synchronous, so calls are offloaded."""

    def __init__(self) -> None:
        self.bucket = settings.minio_bucket
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    async def ensure_ready(self) -> None:
        def _ensure() -> None:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)

        await asyncio.to_thread(_ensure)

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            partial(
                self.client.put_object,
                self.bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
        )

    async def get(self, key: str) -> bytes:
        def _get() -> bytes:
            response = self.client.get_object(self.bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(_get)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.client.remove_object, self.bucket, key)
