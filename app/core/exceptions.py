"""Domain errors and their translation into HTTP responses."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class TEPSIRError(Exception):
    """Base class for expected, domain-level failures."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "tepsir_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(TEPSIRError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(TEPSIRError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class AuthenticationError(TEPSIRError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "authentication_failed"


class PermissionDeniedError(TEPSIRError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "permission_denied"


class ValidationError(TEPSIRError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_error"


class UnsupportedDocumentError(TEPSIRError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_document"


class IndexingError(TEPSIRError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "indexing_failed"


class ServiceUnavailableError(TEPSIRError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(TEPSIRError)
    async def _handle_domain_error(request: Request, exc: TEPSIRError) -> JSONResponse:
        # 5xx are server-side faults, log as errors; 4xx are client faults, log
        # as warnings but still capture the request path for traceability.
        log = logger.error if exc.status_code >= 500 else logger.warning
        log(
            "%s %s -> %d %s: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )
