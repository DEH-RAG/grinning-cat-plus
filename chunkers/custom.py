import asyncio
import json
from typing import List, Iterable, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLSemanticPreservingSplitter,
    RecursiveJsonSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
)
from cat import BaseChunker, EmbeddedBaseChunker

from .semantic_chunker import SemanticChunker as SemanticAnalyzer, MathAwareSemanticChunker as MathAwareSemanticAnalyzer
from .hierarchical_chunker import (
    HierarchicalChunker as HierarchicalAnalyzer,
    MathAwareHierarchicalChunker as MathAwareHierarchicalAnalyzer,
)


class SemanticChunker(EmbeddedBaseChunker):
    def __init__(self, cluster_threshold: float, similarity_threshold: float, max_tokens: int):
        self._cluster_threshold = cluster_threshold
        self._similarity_threshold = similarity_threshold
        self._max_tokens = max_tokens

    @property
    def analyzer(self):
        return SemanticAnalyzer(
            cluster_threshold=self._cluster_threshold,
            similarity_threshold=self._similarity_threshold,
            max_tokens=self._max_tokens
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        await self._ensure_embedder()

        texts = [{"text": doc.page_content, "metadata": doc.metadata} for doc in documents]
        chunks = self.analyzer.chunk(texts)

        return [
            Document(
                page_content=chunk["text"],
                metadata={"source_chunks": chunk.get("metadata", [])}
            )
            for chunk in chunks
        ]


class HTMLSemanticChunker(BaseChunker):
    def __init__(self, headers_to_split_on: List[Tuple[str, str]] | List[List[str]], elements_to_preserve: List[str]):
        self._headers_to_split_on = headers_to_split_on if isinstance(headers_to_split_on[0], tuple) else [
            (header, header) for header in headers_to_split_on
        ]
        self._elements_to_preserve = elements_to_preserve

    @property
    def analyzer(self):
        return HTMLSemanticPreservingSplitter(
            headers_to_split_on=self._headers_to_split_on,
            separators=["\n\n", "\n", ". ", "! ", "? "],
            max_chunk_size=50,
            preserve_images=True,
            preserve_videos=True,
            elements_to_preserve=self._elements_to_preserve,
            denylist_tags=["script", "style", "head"],
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(
            lambda: [chunk for doc in docs for chunk in self.analyzer.split_text(doc.page_content)]
        )


class JSONChunker(BaseChunker):
    def __init__(self, max_chunk_size: int, min_chunk_size: int | None = None):
        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size

    @property
    def analyzer(self):
        return RecursiveJsonSplitter(max_chunk_size=self._max_chunk_size, min_chunk_size=self._min_chunk_size)

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        texts = [json.loads(doc.page_content) for doc in docs]
        metadata = [doc.metadata for doc in docs]
        return await asyncio.to_thread(self.analyzer.create_documents, texts, metadata)


class TokenSpacyChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int, max_length: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_length = max_length

    @property
    def analyzer(self):
        return SpacyTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap, max_length=self._max_length
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(self.analyzer.split_documents, docs)


class TokenNLTKChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int, language: str):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._language = language

    @property
    def analyzer(self):
        return NLTKTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap, language=self._language
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(self.analyzer.split_documents, docs)


class HierarchicalChunker(BaseChunker):
    def __init__(
        self, chunk_size: int, chunk_overlap: int, min_chunk_size: int, max_chunk_size: int, preserve_structure: bool,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._preserve_structure = preserve_structure

    @property
    def analyzer(self):
        return HierarchicalAnalyzer(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_size=self._min_chunk_size,
            max_chunk_size=self._max_chunk_size,
            preserve_structure=self._preserve_structure
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(
            lambda: [chunk for doc in docs for chunk in self.analyzer.chunk_document(doc.page_content, doc.metadata)]
        )


class MathAwareHierarchicalChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int,
        max_chunk_size: int,
        formula_context_window: int,
        preserve_structure: bool,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._formula_context_window = formula_context_window
        self._preserve_structure = preserve_structure

    @property
    def analyzer(self):
        return MathAwareHierarchicalAnalyzer(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_size=self._min_chunk_size,
            max_chunk_size=self._max_chunk_size,
            formula_context_window=self._formula_context_window,
            preserve_structure=self._preserve_structure,
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(
            lambda: [chunk for doc in docs for chunk in self.analyzer.chunk_document(doc.page_content, doc.metadata)]
        )


class MathAwareSemanticChunker(EmbeddedBaseChunker):
    def __init__(
            self,
            cluster_threshold: float,
            similarity_threshold: float,
            max_tokens: int,
            formula_context_boost: float = 0.3,
    ):
        self._cluster_threshold = cluster_threshold
        self._similarity_threshold = similarity_threshold
        self._max_tokens = max_tokens
        self._formula_context_boost = formula_context_boost

    @property
    def analyzer(self):
        return MathAwareSemanticAnalyzer(
            cluster_threshold=self._cluster_threshold,
            similarity_threshold=self._similarity_threshold,
            max_tokens=self._max_tokens,
            formula_context_boost=self._formula_context_boost,
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        await self._ensure_embedder()

        texts = [{"text": doc.page_content, "metadata": doc.metadata} for doc in documents]
        chunks = self.analyzer.chunk(texts)

        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "source_chunks": chunk.get("metadata", []),
                    "has_formula": chunk.get("has_formula", False),
                    "formula_count": chunk.get("formula_count", 0),
                },
            )
            for chunk in chunks
        ]
