"""LLM vision gate: keep recalled images away from text-only chat LLMs.

The core ``multimodal_ingestion`` plugin attaches the recalled memory images to
the agentic workflow as ``image_url`` content parts, but only after asking the
``llm_vision_capable`` hook whether the configured chat LLM can accept image
content. The core's default returns the initial value unchanged (``True`` when
the embedder is multimodal), which means a text-only LLM would receive images
and fail.

This module implements that hook for the LLMs configured by MyPLUS: when the
active chat LLM instance exposes an ``is_multimodal`` capability flag (e.g.
``OpenRouterLLM``, auto-populated from the OpenRouter catalog on save), a
text-only model makes the hook return ``False`` — the recalled images are then
silently skipped and the turn keeps working text-only (the ``[Image]`` text
placeholder stays in the context).

Import-safe: only registers ``@hook`` decorated functions; no top-level side
effects.
"""

from __future__ import annotations

from typing import Any, Optional

from cat import hook
from cat.log import log


def _llm_is_multimodal(cat: Any) -> Optional[bool]:
    """Read the active chat LLM's multimodal capability, when exposed.

    Returns ``True``/``False`` when the instance carries the capability flag
    (set by the MyPLUS LLM configs), ``None`` when it is unknown (LLM from
    another provider without the flag) — the caller then keeps the default.
    """
    llm = getattr(cat, "large_language_model", None)
    if llm is None:
        return None
    flag = getattr(llm, "is_multimodal", None)
    if flag is None:
        return None
    return bool(flag)


@hook(priority=1)
def llm_vision_capable(capable: bool, cat) -> bool:
    """Return False when the active chat LLM is known to be text-only.

    Called by the core before attaching recalled multimodal images to the LLM
    prompt. ``capable`` is ``True`` when the active embedder is multimodal; a
    text-only chat LLM (``is_multimodal=False`` on the instance) forces the
    result to ``False`` so image content is skipped. LLMs without the capability
    flag keep the default unchanged.
    """
    is_multimodal = _llm_is_multimodal(cat)
    if is_multimodal is False:
        log.debug("llm_vision_capable: chat LLM is text-only — recalled images will be skipped")
        return False
    if is_multimodal is True:
        return True
    return capable