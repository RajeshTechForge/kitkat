# src/kitkat/rag/pipeline.py
"""The top-level RAG orchestrator."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kitkat.core.enums import ProviderType, Role
from kitkat.core.models import LLMRequest, LLMResponse, Message
from kitkat.rag.context.assembler import ContextAssembler
from kitkat.rag.context.citation import CitationExtractor
from kitkat.rag.context.prompt import RAGPromptBuilder
from kitkat.rag.core.exceptions import RAGError
from kitkat.rag.ingest.loaders.markdown import MarkdownLoader
from kitkat.rag.ingest.loaders.text import TextLoader
from kitkat.rag.ingest.pipeline import IngestionPipeline, IngestionResult

if TYPE_CHECKING:
    from kitkat.rag.abc.chunker import Chunker
    from kitkat.rag.abc.embedder import EmbeddingProvider
    from kitkat.rag.abc.reranker import Reranker
    from kitkat.rag.abc.retriever import Retriever
    from kitkat.rag.abc.vector_store import VectorStore
    from kitkat.rag.core.models import Document, RetrievalResult
    from kitkat.service.managed import LLMService
    from kitkat.service.router import LLMRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for RAGPipeline behavior."""

    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    use_reranker: bool = True
    max_context_tokens: int = 4096
    context_format_style: str = "numbered"
    include_citation_instructions: bool = True
    default_model: str | None = None
    default_provider: ProviderType | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    cache_retrieval_results: bool = False
    fail_on_retrieval_error: bool = False
    fail_on_ingestion_error: bool = True
    batch_size: int = 100


@dataclass(frozen=True)
class RAGResponse:
    """Complete response from RAGPipeline.ask()."""

    content: str
    sources: list["RetrievalResult"]
    query: str
    context: str
    llm_response: LLMResponse
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


@dataclass(frozen=True)
class RAGStreamChunk:
    """A single chunk from a streamed RAG response."""

    content_delta: str = ""
    sources: list["RetrievalResult"] | None = None
    is_final: bool = False
    usage: Any | None = None


class RAGPipeline:
    """The top-level RAG orchestrator.

    Wires together embedder, chunker, store, retriever, reranker,
    context assembler, and LLM service into a single cohesive pipeline.
    """

    def __init__(
        self,
        *,
        embedder: "EmbeddingProvider",
        store: "VectorStore",
        chunker: "Chunker | None" = None,
        retriever: "Retriever | None" = None,
        reranker: "Reranker | None" = None,
        context_assembler: ContextAssembler | None = None,
        prompt_builder: RAGPromptBuilder | None = None,
        llm_service: "LLMService | LLMRouter | None" = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm_service

        # Lazy defaults for optional components
        if chunker is None:
            from kitkat.rag.chunkers.recursive import RecursiveCharacterChunker

            self._chunker = RecursiveCharacterChunker()
        else:
            self._chunker = chunker

        if retriever is None:
            from kitkat.rag.retrieval.vector import VectorRetriever

            self._retriever = VectorRetriever(embedder=embedder, store=store)
        else:
            self._retriever = retriever

        self._reranker = reranker if config and config.use_reranker else None

        self._context_assembler = context_assembler or ContextAssembler(
            max_context_tokens=config.max_context_tokens if config else 4096,
            format_style=config.context_format_style if config else "numbered",
        )

        self._prompt_builder = prompt_builder or RAGPromptBuilder(
            context_assembler=self._context_assembler,
            include_citation_instructions=config.include_citation_instructions if config else True,
        )

        self._config = config or RAGConfig()

    async def initialize(self) -> None:
        """Initialize all components (embedder, store, reranker, etc.)."""
        await self._embedder.initialize()
        await self._store.initialize()
        if self._reranker:
            # Rerankers might need to load models
            if hasattr(self._reranker, "initialize"):
                await self._reranker.initialize()  # type: ignore[attr-defined]
        logger.info("RAGPipeline initialized.")

    async def shutdown(self) -> None:
        """Shut down all components. Logs and continues on per-component errors."""
        try:
            await self._embedder.shutdown()
        except Exception as exc:
            logger.warning("Error shutting down embedder: %s", exc)

        try:
            await self._store.shutdown()
        except Exception as exc:
            logger.warning("Error shutting down vector store: %s", exc)

        logger.info("RAGPipeline shut down.")

    async def __aenter__(self) -> "RAGPipeline":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.shutdown()

    async def ingest(
        self,
        sources: list[str] | list["Document"],
        *,
        collection: str = "default",
    ) -> IngestionResult:
        """Load, chunk, embed, and store documents."""
        # Determine if sources are paths or Document objects
        are_paths = all(isinstance(s, str) for s in sources)

        pipeline = IngestionPipeline(
            loaders=[TextLoader(), MarkdownLoader()] if are_paths else [],
            transformers=[],
            chunker=self._chunker,
            embedder=self._embedder,
            store=self._store,
            batch_size=self._config.batch_size,
        )

        if are_paths:
            return await pipeline.ingest_sources(sources, collection=collection)  # type: ignore[arg-type]
        else:
            return await pipeline.ingest_documents(sources, collection=collection)  # type: ignore[arg-type]

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        rerank: bool | None = None,
        filter: dict[str, str | int | float | bool] | None = None,
        collection: str = "default",
    ) -> list["RetrievalResult"]:
        """Retrieve and optionally rerank chunks for a query."""
        k = top_k or self._config.retrieval_top_k
        do_rerank = self._config.use_reranker if rerank is None else rerank

        try:
            results = await self._retriever.retrieve(query, top_k=k, filter=filter)

            # Update in-memory keyword retriever if it's part of a hybrid setup
            # (In a real app, this would be handled during ingestion)

            if do_rerank and self._reranker and results:
                rerank_k = top_k or self._config.rerank_top_k
                results = await self._reranker.rerank(query, results, top_k=rerank_k)

            return results
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            if self._config.fail_on_retrieval_error:
                raise RAGError(f"Retrieval failed: {exc}") from exc
            return []

    async def ask(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collection: str = "default",
        model: str | None = None,
        provider: ProviderType | None = None,
        **llm_kwargs: Any,
    ) -> RAGResponse:
        """Retrieve context and generate an answer."""
        start_time = time.monotonic()

        if not self._llm:
            raise RAGError("LLMService not configured for generation. Use retrieve() instead.")

        # 1. Retrieve
        retrieve_start = time.monotonic()
        results = await self.retrieve(query, top_k=top_k, collection=collection)
        retrieval_latency = (time.monotonic() - retrieve_start) * 1000

        # 2. Build Prompt
        messages_tuples = self._prompt_builder.build_messages(query, results)
        messages = [Message(role=Role(r.upper()), content=c) for r, c in messages_tuples]

        context_str = self._context_assembler.assemble(results, query=query)

        # 3. Generate
        gen_start = time.monotonic()
        request = LLMRequest(
            messages=messages,
            model=model or self._config.default_model or "",
            temperature=llm_kwargs.pop("temperature", self._config.temperature),
            max_tokens=llm_kwargs.pop("max_tokens", self._config.max_tokens),
            **llm_kwargs,
        )

        llm_response = (
            await self._llm.complete(request, provider)
            if provider
            else await self._llm.complete(request)
        )
        generation_latency = (time.monotonic() - gen_start) * 1000

        # 4. Extract Citations
        cited_sources = CitationExtractor.extract(llm_response.content, results)

        total_latency = (time.monotonic() - start_time) * 1000

        return RAGResponse(
            content=llm_response.content,
            sources=cited_sources if cited_sources else results,
            query=query,
            context=context_str,
            llm_response=llm_response,
            retrieval_latency_ms=retrieval_latency,
            generation_latency_ms=generation_latency,
            total_latency_ms=total_latency,
        )

    async def ask_stream(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collection: str = "default",
        model: str | None = None,
        provider: ProviderType | None = None,
        **llm_kwargs: Any,
    ) -> AsyncIterator[RAGStreamChunk]:
        """Retrieve context and stream the LLM response."""
        if not self._llm:
            raise RAGError("LLMService not configured for generation.")

        results = await self.retrieve(query, top_k=top_k, collection=collection)

        # Yield sources first so the client can display them immediately
        yield RAGStreamChunk(sources=results)

        messages_tuples = self._prompt_builder.build_messages(query, results)
        messages = [Message(role=Role(r.upper()), content=c) for r, c in messages_tuples]

        request = LLMRequest(
            messages=messages,
            model=model or self._config.default_model or "",
            temperature=llm_kwargs.pop("temperature", self._config.temperature),
            max_tokens=llm_kwargs.pop("max_tokens", self._config.max_tokens),
            **llm_kwargs,
        )

        # Stream from LLM
        stream_gen = self._llm.stream(request, provider) if provider else self._llm.stream(request)

        async for chunk in stream_gen:
            is_final = chunk.is_final
            yield RAGStreamChunk(
                content_delta=chunk.content,
                is_final=is_final,
                usage=chunk.usage if is_final else None,
            )
