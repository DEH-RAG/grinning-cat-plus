import base64
import os
from typing import Any, Dict, List
import httpx
import requests
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import FastEmbedEmbeddings
from cat import Embeddings, MultimodalEmbeddings
from cat.utils import retrieve_image

from cat import log

_EMBEDDERS_MODELS_CACHE = {}


class CustomFastEmbedEmbeddings(Embeddings):
    """Wrapper for FastEmbedEmbeddings that inherits from cat.Embeddings.

    FastEmbedEmbeddings from langchain_community inherits from langchain_core.embeddings.Embeddings,
    which is a sibling (not parent) of cat.Embeddings. The factory validation in
    BaseFactoryConfigModel.get_from_config() requires issubclass(pyclass, cat.Embeddings),
    so a wrapper is needed to pass the check.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en",
        max_length: int = 512,
        doc_embed_type: str = "passage",
        cache_dir: str = "cat/data/models/fast_embed",
    ):
        self._inner = FastEmbedEmbeddings(
            model_name=model_name,
            max_length=max_length,
            doc_embed_type=doc_embed_type,
            cache_dir=cache_dir,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)


class CustomOpenAIEmbeddings(Embeddings):
    """Use OpenAI-compatible API as embedder (like llama-cpp-python)."""
    def __init__(self, url: str, model: str, api_key: str | None = None):
        self.url = os.path.join(url, "v1/embeddings")
        self.model = model
        self.api_key = api_key

    @property
    def headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API expects JSON payload, not form data
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"input": texts, "model": self.model},
            timeout=300,
        )
        response.raise_for_status()

        to_return = [e["embedding"] for e in response.json()["data"]]
        return to_return

    def embed_query(self, text: str) -> List[float]:
        # OpenAI API expects JSON payload, not form data
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"input": text, "model": self.model},
            timeout=300,
        )
        response.raise_for_status()

        to_return = response.json()["data"][0]["embedding"]
        return to_return


class CustomOllamaEmbeddings(Embeddings):
    """Use Ollama to serve embedding models."""
    def __init__(self, base_url: str, model: str):
        self.url = os.path.join(base_url, "api/embeddings")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't support batch processing, so we need to process one by one
        embeddings = []
        for text in texts:
            ret = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=300.0)
            ret.raise_for_status()
            embeddings.append(ret.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        ret = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=300.0)
        ret.raise_for_status()
        return ret.json()["embedding"]


class CustomJinaEmbedder(Embeddings):
    """Use Jina AI to serve embedding models."""
    def __init__(self, base_url: str, model: str, api_key: str, task: str = "text-matching"):
        self.url = os.path.join(base_url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task

    def _embed(self, texts: List[str]) -> List[List[float]]:
        ret = httpx.post(
            self.url,
            data={"model": self.model, "input": texts, "task": self.task},
            timeout=300.0,
            headers=self.headers,
        )
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class Qwen3LocalEmbeddings(Embeddings):
    """
    Local Qwen3 embeddings using HuggingFace Sentence Transformers.
    Best for: Full control, no external dependencies, offline usage
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the model"""
        global _EMBEDDERS_MODELS_CACHE

        model = _EMBEDDERS_MODELS_CACHE.get("Qwen3LocalEmbeddings", {}).get(self.model_name)
        if model is None:
            model = SentenceTransformer(self.model_name, trust_remote_code=True)
            _EMBEDDERS_MODELS_CACHE.setdefault("Qwen3LocalEmbeddings", {})[self.model_name] = model
        return model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        model = self._load_model()
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


class Qwen3OllamaEmbeddings(Embeddings):
    """
    Qwen3 embeddings via Ollama.
    Best for: Easy local deployment, minimal setup
    """
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)


class Qwen3DeepInfraEmbeddings(Embeddings):
    """
    Qwen3 embeddings via DeepInfra API (OpenAI-compatible).
    Best for: Production deployment, no GPU required, pay-as-you-go
    """
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from DeepInfra"""
        if not self.api_key:
            raise ValueError("DeepInfra API key is required")

        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "input": texts,
                    "model": self.model_name,
                    "encoding_format": "float"
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            sorted_embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_embeddings]
        except requests.RequestException as e:
            raise RuntimeError(f"DeepInfra embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        # DeepInfra supports batch processing
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embeddings([text])[0]


class Qwen3TEIEmbeddings(Embeddings):
    """
    Qwen3 embeddings via Text Embeddings Inference (self-hosted).
    Best for: High-throughput production, full control, optimized inference
    """
    def __init__(self, base_url: str):
        self.base_url = base_url

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from TEI server"""
        try:
            response = requests.post(
                f"{self.base_url}/embed",
                json={"inputs": texts},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"TEI embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embeddings([text])[0]


class CustomJinaMultimodalEmbedder(MultimodalEmbeddings):
    """Use Jina AI to serve embedding multimodal models."""
    def __init__(self, base_url: str, model: str, api_key: str, task: str = "text-matching"):
        self.url = os.path.join(base_url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task

    def _embed(
        self,
        texts: List[str] | None = None,
        images: List[str | bytes] | None = None
    ) -> List[List[float]]:
        def parse_image(image: str | bytes) -> str:
            if isinstance(image, bytes):
                return base64.b64encode(image).decode("utf-8")
            image = retrieve_image(image)
            # remove "data:image/...;base64," prefix if present
            if image is not None and image.startswith("data:image"):
                image = image.split(",", 1)[1]
            return image

        payload = (
            [{"text": t} for t in texts] if texts else []
        ) + (
            [{"image": parse_image(i)} for i in images if i] if images else []
        )

        if not payload:
            return []

        ret = httpx.post(
            self.url,
            json={"model": self.model, "input": payload, "task": self.task},
            headers=self.headers,
            timeout=300.0,
        )
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed(texts=[text])[0]

    def embed_image(self, image: str | bytes) -> List[float]:
        return self._embed(images=[image])[0]

    def embed_images(self, images: List[str | bytes]) -> List[List[float]]:
        return self._embed(images=images)


class JinaCLIPEmbeddings(MultimodalEmbeddings):
    """
    Jina CLIP v2 multimodal embeddings.
    Handles both text and images in same vector space.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def _get_embeddings(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Get embeddings from Jina API.

        Args:
            inputs: List of {"text": str} or {"image": bytes/url}
        """
        if not self.api_key:
            raise ValueError("Jina API key required")

        try:
            # Prepare input for Jina API
            prepared_inputs = []
            for inp in inputs:
                if "text" in inp:
                    prepared_inputs.append({"text": inp["text"]})
                elif "image" in inp:
                    # Handle image bytes or URL
                    tmp = inp["image"]
                    if isinstance(inp["image"], bytes):
                        img_b64 = base64.b64encode(inp["image"]).decode()
                        tmp = f"data:image/png;base64,{img_b64}"

                    prepared_inputs.append({"image": tmp})

            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model_name,
                    "input": prepared_inputs
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            return [item["embedding"] for item in data["data"]]

        except requests.RequestException as e:
            raise RuntimeError(f"Jina embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents"""
        inputs = [{"text": text} for text in texts]
        return self._get_embeddings(inputs)

    def embed_query(self, text: str) -> List[float]:
        """Embed single text query"""
        return self._get_embeddings([{"text": text}])[0]

    def embed_image(self, image: bytes) -> List[float]:
        """Embed single image"""
        return self._get_embeddings([{"image": image}])[0]

    def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """Embed multiple images"""
        inputs = [{"image": img} for img in images]
        return self._get_embeddings(inputs)


class CustomVllmMultimodalEmbedder(MultimodalEmbeddings):
    """Multimodal embeddings via vLLM's OpenAI-compatible /v1/embeddings endpoint.

    Uses the standard OpenAI string-list ``input`` form: each entry is either a
    plain text string or an image ``data:<mime>;base64,<...>`` URI. vLLM returns
    one embedding per input entry (aligned by ``index``), which is required by
    the chunkers/semantic chunker that call ``embed_documents`` on many chunks.

    NOTE: the chat-message form (``input: [{role, content:[{type:...}]}]``)
    aggregates the whole request into a SINGLE pooled vector, so it is NOT used
    here — it would break batch embedding.
    """

    # magic-byte prefixes -> mime type, used to build data URIs for bytes input
    _MAGIC_TO_MIME = (
        (b"\x89PNG", "image/png"),
        (b"\xff\xd8", "image/jpeg"),
        (b"GIF8", "image/gif"),
        (b"RIFF", "image/webp"),
        (b"BM", "image/bmp"),
    )

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        task: str | None = None,
        timeout: float = 300.0,
    ):
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task
        self.timeout = timeout

    @staticmethod
    def _sniff_mime(data: bytes) -> str:
        for magic, mime in CustomVllmMultimodalEmbedder._MAGIC_TO_MIME:
            if data.startswith(magic):
                return mime
        return "image/png"

    def _to_data_uri(self, image: str | bytes) -> str:
        """Return a full ``data:<mime>;base64,<...>`` URI for image input."""
        if isinstance(image, bytes):
            return f"data:{self._sniff_mime(image)};base64,"\
                   f"{base64.b64encode(image).decode('utf-8')}"
        uri = retrieve_image(image)
        if isinstance(uri, bytes):
            return self._to_data_uri(uri)
        if uri is None:
            raise ValueError(f"Unable to read image: {image!r}")
        uri = str(uri)
        if uri.startswith("data:"):
            return uri
        # treat as raw base64 (e.g. retrieve_image returned bare base64)
        return f"data:image/png;base64,{uri}"

    def _embed(
        self,
        items: List[Dict[str, Any]],
    ) -> List[List[float]]:
        if not items:
            return []

        payload = {"model": self.model, "input": []}
        for it in items:
            if "text" in it:
                payload["input"].append(it["text"])
            elif "image" in it:
                payload["input"].append(self._to_data_uri(it["image"]))

        ret = httpx.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        ret.raise_for_status()
        data = ret.json().get("data")
        if not data:
            raise RuntimeError(f"vLLM embedding returned no data: {ret.text[:200]}")
        # vLLM returns one entry per input, ordered by index
        return [entry["embedding"] for entry in sorted(data, key=lambda e: e.get("index", 0))]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed([{"text": t} for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._embed([{"text": text}])[0]

    def embed_image(self, image: str | bytes) -> List[float]:
        return self._embed([{"image": image}])[0]

    def embed_images(self, images: List[str | bytes]) -> List[List[float]]:
        return self._embed([{"image": img} for img in images])
