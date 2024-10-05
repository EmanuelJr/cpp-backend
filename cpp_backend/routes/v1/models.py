from fastapi import Body, Depends, APIRouter

from cpp_backend.utils.middlewares import authenticate
from cpp_backend.schemas import openai_v1_tag, cpp_tag
from cpp_backend.schemas.model import (
    ModelList,
    LlamaModelSettings,
    ModelUnloadRequest,
    WhisperModelSettings,
)
from cpp_backend.models import llama_models, whisper_models

router = APIRouter(
    prefix="/models",
)


# Llama models
@router.post(
    "/llama",
    summary="Loads a Llama model",
    dependencies=[Depends(authenticate)],
    tags=[cpp_tag],
)
async def load_models(body: LlamaModelSettings = Body()) -> ModelList:
    llama_models.load(body)

    return {
        "object": "list",
        "data": [
            {
                "id": body.alias,
                "object": "model",
                "owned_by": "local",
                "permissions": [],
            }
        ],
    }


@router.delete(
    "/llama",
    summary="Unloads a Llama model",
    dependencies=[Depends(authenticate)],
    tags=[cpp_tag],
)
async def unload_model(body: ModelUnloadRequest = Body()) -> ModelList:
    llama_models.unload(body.alias)

    return {
        "object": "list",
        "data": [
            {
                "id": body.alias,
                "object": "model",
                "owned_by": "local",
                "permissions": [],
            }
        ],
    }


@router.post(
    "/whisper",
    summary="Loads a Whisper model",
    dependencies=[Depends(authenticate)],
    tags=[cpp_tag],
)
async def load_models(body: WhisperModelSettings = Body()) -> ModelList:
    whisper_models.load(body)

    return {
        "object": "list",
        "data": [
            {
                "id": body.alias,
                "object": "model",
                "owned_by": "local",
                "permissions": [],
            }
        ],
    }


@router.delete(
    "/whisper",
    summary="Unloads a Whisper model",
    dependencies=[Depends(authenticate)],
    tags=[cpp_tag],
)
async def unload_model(body: ModelUnloadRequest = Body()) -> ModelList:
    whisper_models.unload(body.alias)

    return {
        "object": "list",
        "data": [
            {
                "id": body.alias,
                "object": "model",
                "owned_by": "local",
                "permissions": [],
            }
        ],
    }


# Agnostic to any model
@router.get(
    "",
    summary="Models",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def get_models() -> ModelList:
    return {
        "object": "list",
        "data": [
            {
                "id": alias,
                "object": "model",
                "owned_by": "local",
                "permissions": [],
            }
            for alias in llama_models.list() + whisper_models.list()
        ],
    }
