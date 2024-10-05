import io
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, File, HTTPException, status
from cpp_backend.models import whisper_models
from cpp_backend.schemas import openai_v1_tag

router = APIRouter(
    prefix="/audio",
)


@router.post("/transcriptions", summary="Transcribe audio", tags=[openai_v1_tag])
def transcription(
    file: Annotated[bytes, File()],
    model: Annotated[str, Body()],
    prompt: Annotated[str, Body()] = None,
    response_format: Annotated[
        Literal["json", "text", "srt", "verbose_json", "vtt"], Body()
    ] = "json",
    temperature: Annotated[float, Body()] = 0.8,
    language: Annotated[str, Body()] = "en",
) -> Any:
    whisper = whisper_models.get(model)
    if whisper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    return whisper.transcribe(
        io.BytesIO(file), prompt, response_format, temperature, language
    )


@router.post("/translations", description="Translate audio", tags=[openai_v1_tag])
def translation(
    file: Annotated[bytes, File()],
    model: Annotated[str, Body()],
    prompt: Annotated[str, Body()] = None,
    response_format: Annotated[
        Literal["json", "text", "srt", "verbose_json", "vtt"], Body()
    ] = "json",
    temperature: Annotated[float, Body()] = 0.8,
) -> Any:
    whisper = whisper_models.get(model)
    return whisper.translate(io.BytesIO(file), prompt, response_format, temperature)
