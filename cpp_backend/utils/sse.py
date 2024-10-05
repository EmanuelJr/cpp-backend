import json
import typing
from typing import Iterator

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import iterate_in_threadpool
from fastapi import Request


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream[typing.Any],
    iterator: Iterator[typing.Any],
    on_complete: typing.Optional[typing.Callable[[], None]] = None,
):
    async with inner_send_chan:
        try:
            async for chunk in iterate_in_threadpool(iterator):
                await inner_send_chan.send(dict(data=json.dumps(chunk)))
                if await request.is_disconnected():
                    raise anyio.get_cancelled_exc_class()()

            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            print("disconnected")
            with anyio.move_on_after(1, shield=True):
                print(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
        finally:
            if on_complete:
                on_complete()
