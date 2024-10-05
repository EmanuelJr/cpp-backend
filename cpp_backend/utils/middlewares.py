from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

import cpp_backend.settings as settings

bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate(
    authorization: Optional[str] = Depends(bearer_scheme),
):
    # Skip API key check if it's not set in settings
    if settings.api_key is None:
        return True

    # check bearer credentials against the api_key
    if authorization and authorization.credentials == settings.api_key:
        # api key is valid
        return authorization.credentials

    # raise http error 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
    )
