"""Domain errors and their translation into HTTP responses."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


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


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(TEPSIRError)
    async def _handle_domain_error(_: Request, exc: TEPSIRError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "code": exc.code},
        )
