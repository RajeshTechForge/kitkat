"""Custom exception hierarchy for KitKat."""

from __future__ import annotations


class KitkatError(Exception):
    """Base exception for all Nexus errors."""

    def __init__(
        self,
        message: str,
        code: str = "NEXUS_ERROR",
        details: dict | None = None,
        status_code: int = 500,
    ):
        self.message = message
        self.code = code
        self.details: dict | None = details
        self.status_code = status_code
        super().__init__(self.message)
