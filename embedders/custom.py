import base64
import json
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

    Uses vLLM's batch-chat request form: ``input`` is a list of conversations,
    one per item, each a single ``user`` message whose ``content`` is typed
    parts (``{"type": "text", "text": ...}`` / ``{"type": "image_url",
    "image_url": {"url": "data:...;base64,<...>"}}``). vLLM returns ONE vector
    per conversation (aligned by ``index``) and decodes images as real images
    (vision tokens) — required by the chunkers that call ``embed_documents`` on
    many chunks and by ``embed_images`` for PDF-extracted images.

    NOTE: the plain string-list ``input`` form is NOT usable for images: vLLM
    tokenizes the whole base64 URI as text, blowing the context window, and the
    single-conversation chat form pools the whole request into one vector.
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
        max_image_tokens: int = 2048,
        query_prefix: str = "Query: ",
        document_prefix: str = "Document: ",
    ):
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task
        self.timeout = timeout
        # Max image tokens the embedding model accepts (Jina v5 omni: ~2048).
        # Other multimodal models may accept more; used by the downscaler.
        self.max_image_tokens = max_image_tokens
        # Retrieval-side prefixes (Jina v5 defaults). Queries vs documents are
        # embedded differently by retrieval models; prepending the matching side
        # prefix aligns with encode_query()/encode_document(). Overridable for
        # other models (e.g. "Passage: " for document side).
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        # conservative budget: leave headroom below the model's 32768-token context
        self._max_input_tokens = 30000

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """Rough token estimate: ~3 chars per token (conservative)."""
        return max(1, len(text) // 3)

    def _split_batches(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split items into request batches that stay within the context window.

        Images go one per request because their token cost scales with pixels and
        cannot be estimated cheaply; texts are grouped up to the token budget.
        """
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0

        for it in items:
            if "image" in it:
                if current:
                    batches.append(current)
                    current, current_tokens = [], 0
                batches.append([it])
                continue

            tokens = self._estimate_text_tokens(it.get("text", ""))
            if current and current_tokens + tokens > self._max_input_tokens:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(it)
            current_tokens += tokens

        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _sniff_mime(data: bytes) -> str:
        for magic, mime in CustomVllmMultimodalEmbedder._MAGIC_TO_MIME:
            if data.startswith(magic):
                return mime
        return "image/png"

    def _image_grid_tokens(self, width: int, height: int) -> int:
        """Estimate the number of image tokens the Qwen3VL processor will produce.

        Replicates the processor's smart_resize: each side is rounded to a
        multiple of patch_size(16) * merge_size(2) = 32, then divided by 16 to
        get the grid dims; tokens = grid_h * grid_w.
        """
        grid_w = round(width / 32) * 2
        grid_h = round(height / 32) * 2
        return grid_w * grid_h

    def _resize_image_if_needed(self, image: bytes) -> bytes:
        """Downscale an image so its processor token budget fits the model.

        The Jina v5 omni *embedding* model rejects images whose processor grid
        exceeds ~2048 image tokens (observed: <=2024 OK, >=2052 FAIL), regardless
        of aspect ratio. Preserve aspect ratio and shrink just enough to land
        under ``self.max_image_tokens`` — wide/short or tall/narrow images keep
        more resolution than forcing a square cap. Falls back to the original
        bytes if PIL is unavailable or the decode fails.
        """
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image))
            img.load()
            w, h = img.size
            if self._image_grid_tokens(w, h) <= self.max_image_tokens:
                return image

            # binary search the largest scale keeping tokens under the budget
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                nw = max(1, round(w * mid))
                nh = max(1, round(h * mid))
                if self._image_grid_tokens(nw, nh) <= self.max_image_tokens:
                    lo = mid
                else:
                    hi = mid

            nw = max(1, round(w * lo))
            nh = max(1, round(h * lo))
            img = img.resize((nw, nh), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001 - best-effort resize
            log.debug(f"VLLM_EMBEDDINGS image resize skipped: {exc}")
            return image

    def _to_data_uri(self, image: str | bytes) -> str:
        """Return a full ``data:<mime>;base64,<...>`` URI for image input."""
        if isinstance(image, bytes):
            image = self._resize_image_if_needed(image)
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

    @staticmethod
    def _prefixed(text: str, prefix: str) -> str:
        """Join a prefix and the text ensuring they are space-separated."""
        text = text.strip()
        if not prefix or not text:
            return text
        return prefix.rstrip() + " " + text if text else prefix

    def _embed(
        self,
        items: List[Dict[str, Any]],
        prefix: str = "",
    ) -> List[List[float]]:
        if not items:
            return []

        # vLLM batch-chat form: input = one conversation per item; each
        # conversation is a single message whose content is typed parts
        # (text / image_url). This returns ONE vector per conversation and
        # decodes images as real images (vision tokens), not as base64 text.
        conversations = []
        for it in items:
            content = []
            if "text" in it:
                text = it["text"] if it["text"] is not None else ""
                # vLLM rejects empty prompts; placeholder keeps alignment
                text = text if text.strip() else " "
                content.append({"type": "text", "text": self._prefixed(text, prefix)})
            elif "image" in it:
                # The retrieval prefix (Query:/Document:) applies to media too:
                # prepend it as a text part right before the image (the chat
                # template renders it next to the media placeholder).
                if prefix:
                    content.append({"type": "text", "text": prefix.rstrip()})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._to_data_uri(it["image"])},
                })
            conversations.append({"role": "user", "content": content})

        payload = {"model": self.model, "input": [[c] for c in conversations]}

        # Debug: log the exact request dict, shortening image data URIs to ~20 chars
        debug_input = []
        for conv in conversations:
            debug_content = []
            for part in conv["content"]:
                if part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    head, _, b64 = url.partition(";base64,")
                    if len(b64) > 20:
                        b64 = b64[:20] + f"...({len(b64)} b64 chars)"
                    debug_content.append({"type": "image_url",
                                          "image_url": {"url": f"{head};base64,{b64}"}})
                else:
                    debug_content.append(part)
            debug_input.append([{"role": "user", "content": debug_content}])
        log.debug(f"VLLM_EMBEDDINGS request to {self.url}: "
                  f"{json.dumps({'model': payload['model'], 'input': debug_input}, default=str)}")

        ret = httpx.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        if ret.status_code != 200:
            raise RuntimeError(
                f"vLLM embedding failed with status {ret.status_code}: {ret.text[:500]}"
            )
        data = ret.json().get("data")
        if not data:
            raise RuntimeError(f"vLLM embedding returned no data: {ret.text[:200]}")
        # vLLM returns one entry per conversation, ordered by index
        return [entry["embedding"] for entry in sorted(data, key=lambda e: e.get("index", 0))]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for batch in self._split_batches([{"text": t} for t in texts]):
            results.extend(self._embed(batch, prefix=self.document_prefix))
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed([{"text": text}], prefix=self.query_prefix)[0]

    def embed_image(self, image: str | bytes) -> List[float]:
        return self._embed([{"image": image}], prefix=self.document_prefix)[0]

    def embed_images(self, images: List[str | bytes]) -> List[List[float]]:
        results = []
        for batch in self._split_batches([{"image": img} for img in images]):
            results.extend(self._embed(batch, prefix=self.document_prefix))
        return results
