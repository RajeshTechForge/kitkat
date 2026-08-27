"""Pydantic V2 schemas for the RAG API boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from kitkat.rag.core.models import Document


class DocumentSchema(BaseModel):
    """Pydantic schema for document ingestion requests."""

    content: str = Field(..., min_length=1)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    source: str = ""
    content_type: str = "text/plain"

    @field_validator("content")
    @classmethod
    def content_not_whitespace(cls, v: str) -> str:
        """Validate content is not empty or whitespace."""
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v

    def to_domain(self) -> Document:
        """Convert to domain dataclass."""
        return Document(
            content=self.content,
            metadata=self.metadata,
            source=self.source,
            content_type=self.content_type,
        )
