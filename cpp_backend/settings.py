from os import environ as env
from typing import Optional

server_host = env.get("HOST", "127.0.0.1")
server_port: int = int(env.get("PORT", 8000))

api_key: Optional[str] = env.get("API_KEY")
