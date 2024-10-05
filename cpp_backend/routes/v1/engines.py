from typing import Annotated
from fastapi import Depends, APIRouter, Path, Request
import llama_cpp
from cpp_backend.schemas.completion import CreateCompletionRequest
from cpp_backend.utils.middlewares import authenticate
from cpp_backend.routes.v1.completions import create_completion
from cpp_backend.schemas import openai_v1_tag

router = APIRouter(
    prefix="/engines",
)


@router.post(
    "/{model}/completions",
    summary="Engines Completion",
    dependencies=[Depends(authenticate)],
    tags=[openai_v1_tag],
)
async def create_engines_completion(
    request: Request,
    body: CreateCompletionRequest,
    model: Annotated[str, Path(title="Model name")],
) -> llama_cpp.Completion:
    params = {**body.model_dump(), model: model}

    return await create_completion(request, params)
