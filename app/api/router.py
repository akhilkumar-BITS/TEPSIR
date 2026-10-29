from fastapi import APIRouter

from app.api.v1 import assistant, auth, documents, search, users

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(assistant.router)
