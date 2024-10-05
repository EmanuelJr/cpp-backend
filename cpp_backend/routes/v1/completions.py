import contextlib
from typing import Iterator, Union
import anyio
from fastapi import Depends, APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
import llama_cpp
from sse_starlette import EventSourceResponse

from cpp_backend.models import logit_bias_tokens_to_input_ids, llama_models
from cpp_backend.schemas.completion import CreateCompletionRequest
from cpp_backend.utils.middlewares import authenticate
from cpp_backend.schemas import openai_v1_tag
from cpp_backend.utils.sse import get_event_publisher
from cpp_backend.models import llama_models

router = APIRouter(
    prefix="/completions",
)


@router.post(
    "",
    summary="Completion",
    dependencies=[Depends(authenticate)],
    response_model=Union[
        llama_cpp.CreateCompletionResponse,
        str,
    ],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {"$ref": "#/components/schemas/CreateCompletionResponse"}
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True. "
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest,
) -> llama_cpp.Completion:
    exit_stack = contextlib.ExitStack()
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    llama = llama_models.get(body.model)

    if llama is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    exclude = {
        "n",
        "best_of",
        "logit_bias_type",
        "user",
        "min_tokens",
    }
    kwargs = body.model_dump(exclude=exclude)

    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    if body.min_tokens > 0:
        _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
            [llama_cpp.MinTokensLogitsProcessor(body.min_tokens, llama.token_eos())]
        )
        if "logits_processor" not in kwargs:
            kwargs["logits_processor"] = _min_tokens_logits_processor
        else:
            kwargs["logits_processor"].extend(_min_tokens_logits_processor)

    iterator_or_completion: Union[
        llama_cpp.CreateCompletionResponse,
        Iterator[llama_cpp.CreateCompletionStreamResponse],
    ] = await run_in_threadpool(llama, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.CreateCompletionStreamResponse]:
            yield first_response
            yield from iterator_or_completion
            exit_stack.close()

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
                on_complete=exit_stack.close,
            ),
            sep="\n",
        )
    else:
        return iterator_or_completion
