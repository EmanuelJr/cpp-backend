from typing import List, Optional, Union
from pydantic import BaseModel, Field

from cpp_backend.schemas import model_field


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = model_field
    input: Union[str, List[str]] = Field(description="Input to be embedded.")
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The quick brown fox jumps over the lazy dog.",
                }
            ]
        }
    }
