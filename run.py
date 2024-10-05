import argparse
import logging
import uvicorn

import cpp_backend.settings as settings
from cpp_backend.app import create_app

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="cpp-backend",
        description="An OpenAI Completions API compatible server for locally running .cpp backends",
    )
    parser.add_argument(
        "host",
        nargs="?",
        help="Host to run the server on (default: 127.0.0.1:8000)",
        default="127.0.0.1:8000",
    )
    args = parser.parse_args()

    server_host, server_port = args.host.split(":")
    settings.server_host = server_host
    settings.server_port = int(server_port)


def main():
    parse_args()

    app = create_app()

    logger.info(f"Starting server at {settings.server_host}:{settings.server_port}")
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)


if __name__ == "__main__":
    main()
