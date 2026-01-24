import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log incoming requests and outgoing responses.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            The HTTP response
        """
        # Generate request ID for tracking
        request_id = id(request)

        # Log request
        logger.info(
            f"Request started | id={request_id} | method={request.method} | "
            f"path={request.url.path} | client={request.client.host if request.client else 'unknown'}"
        )

        # Process request and measure time
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Request completed | id={request_id} | method={request.method} | "
                f"path={request.url.path} | status={response.status_code} | "
                f"duration={process_time:.3f}s"
            )

            # Add process time header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed | id={request_id} | method={request.method} | "
                f"path={request.url.path} | error={str(e)} | "
                f"duration={process_time:.3f}s",
                exc_info=True,
            )
            raise
