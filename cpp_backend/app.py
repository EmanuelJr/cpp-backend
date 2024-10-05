from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cpp_backend.schemas import cpp_tag
from cpp_backend.routes import v1


def create_app():
    app = FastAPI(title="cpp Backend", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(v1.router)

    # Health endpoint
    @app.get(
        "/health",
        summary="Health check",
        tags=[cpp_tag],
    )
    async def health():
        return {"status": "ok"}

    return app
