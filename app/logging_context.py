"""Propagates the model of the in-flight Mantle request through the async
call stack (including into httpx's own internal logging) without having to
thread it through every function signature or log call.
"""

import contextvars
import logging

current_model: contextvars.ContextVar[str] = contextvars.ContextVar("current_model", default="-")


class ModelLogFilter(logging.Filter):
    """Attach to a Handler (not a Logger) so it runs for every record that
    reaches it, regardless of which logger emitted it — this is what makes
    it apply to httpx's own "HTTP Request: ..." log lines too."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.model = current_model.get()
        return True
