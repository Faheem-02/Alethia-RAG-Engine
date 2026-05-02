"""Logging utilities for backend observability.

Responsibilities:
- Provide a consistent logger factory across modules.
- Standardize log formatting and levels.
- Keep logging setup centralized.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Stub: Return a module-level logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
