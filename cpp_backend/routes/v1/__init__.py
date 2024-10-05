from fastapi import APIRouter

from cpp_backend.routes.v1 import (
    audio,
    models,
    completions,
    chat,
    embeddings,
    engines,
)

router = APIRouter(
    prefix="/v1",
)

router.include_router(models.router)
router.include_router(completions.router)
router.include_router(chat.router)
router.include_router(embeddings.router)
router.include_router(engines.router)
router.include_router(audio.router)
