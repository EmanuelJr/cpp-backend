from fastapi import Depends, APIRouter, HTTPException, status

from cpp_backend.schemas.embedding import CreateEmbeddingRequest
from cpp_backend.utils.middlewares import authenticate
from cpp_backend.schemas import openai_v1_tag
from cpp_backend.models import llama_models

router = APIRouter(
    prefix="/embeddings",
)


@router.post(
    "",
    summary="Embedding",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def create_embedding(request: CreateEmbeddingRequest):
    model = llama_models.get(request.model)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    model.create_embedding(**request.model_dump(exclude={"user"}))
