from typing import Any
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class CustomOpenAI(ChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(model_kwargs={}, **kwargs)


class CustomOllama(ChatOllama):
    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("base_url", "").endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        super().__init__(**kwargs)


class OpenRouterLLM(CustomOpenAI):
    """ChatOpenAI bound to OpenRouter's OpenAI-compatible API.

    The ``LLMOpenRouterConfig`` settings carry extra fields that LangChain's
    ``ChatOpenAI`` does not know about (``is_multimodal``, ``max_token_context``,
    the per-1M-token costs used for accounting, ...). This class pops those fields
    out of the constructor kwargs before delegating to ``ChatOpenAI`` and keeps them
    as attributes, so the rest of the framework (multimodal dispatch, context-window
    splitting, accounting) can read the model capabilities at runtime.
    """

    _EXTRA_FIELDS = (
        "is_multimodal",
        "max_token_context",
        "max_completion_tokens",
        "input_modalities",
        "prompt_cost_per_1m",
        "completion_cost_per_1m",
        "input_cache_read_cost_per_1m",
        "request_cost",
        "referer",
        "site_title",
    )

    def __init__(self, **kwargs: Any) -> None:
        # Pop model-capability fields; keep them as attributes for runtime use.
        for field in self._EXTRA_FIELDS:
            setattr(self, field, kwargs.pop(field, None))

        # OpenRouter attribution headers (optional, recommended by OpenRouter).
        referer = kwargs.pop("referer", None)
        site_title = kwargs.pop("site_title", None)
        default_headers = dict(kwargs.pop("default_headers", None) or {})
        if referer:
            default_headers["HTTP-Referer"] = referer
        if site_title:
            default_headers["X-Title"] = site_title
        if default_headers:
            kwargs["default_headers"] = default_headers

        kwargs.setdefault("base_url", OPENROUTER_BASE_URL)
        super().__init__(model_kwargs={}, **kwargs)
