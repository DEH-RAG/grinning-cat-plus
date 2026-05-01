import dataclasses
import os
import tempfile
from typing import Any, Iterator

import numpy as np
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document
from langchain_unstructured import UnstructuredLoader


class UnstructuredParser(BaseBlobParser):
    """A `BaseBlobParser` backed by `langchain_unstructured.UnstructuredLoader`.

    Differently from the previous implementation that depended on
    `langchain_community.document_loaders.UnstructuredFileLoader` subclasses,
    this parser now relies on the unified `UnstructuredLoader` which auto-detects
    the file type (via the underlying `unstructured.partition.auto.partition`)
    and supports every MIME type for which the corresponding optional
    `unstructured` extras are installed.

    The constructor accepts arbitrary keyword arguments that are forwarded
    verbatim to `UnstructuredLoader` and, in turn, to `partition()`. Sensible
    defaults oriented to multimodal extraction (`hi_res` strategy, image and
    table extraction) are provided when no kwargs are passed.
    """

    _DEFAULT_KWARGS = {
        "strategy": "hi_res",
        "extract_images_in_pdf": True,
        "infer_table_structure": True,
        "extract_image_block_types": ["Image", "Table"],
    }

    def __init__(self, **partition_kwargs: Any):
        merged: dict[str, Any] = dict(self._DEFAULT_KWARGS)
        merged.update(partition_kwargs)
        # `partition_via_api` is forced to False so we always run the local
        # unstructured stack, regardless of any UNSTRUCTURED_API_KEY being set
        # in the environment.
        merged.setdefault("partition_via_api", False)
        self._partition_kwargs = merged

    @property
    def partition_kwargs(self) -> dict[str, Any]:
        return dict(self._partition_kwargs)

    @staticmethod
    def _serialize_metadata_value(value: Any) -> Any:
        """Convert non-serializable values to JSON-compatible format.

        `UnstructuredLoader.lazy_load()` already produces metadata as plain
        dictionaries (via `element.to_dict()`), so most values are JSON ready.
        This helper is kept as a defensive normalization step for edge cases
        (numpy scalars, dataclasses, pydantic models, slot-only classes...).
        """
        if value is None:
            return None
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return [UnstructuredParser._serialize_metadata_value(item) for item in value]
        if isinstance(value, list):
            return [UnstructuredParser._serialize_metadata_value(item) for item in value]
        if isinstance(value, dict):
            return {k: UnstructuredParser._serialize_metadata_value(v) for k, v in value.items()}
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return UnstructuredParser._serialize_metadata_value(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict") and callable(value.dict) and not isinstance(value, dict):
            try:
                return UnstructuredParser._serialize_metadata_value(value.dict())
            except Exception:
                pass
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            try:
                return UnstructuredParser._serialize_metadata_value(dataclasses.asdict(value))  # type: ignore[arg-type]
            except Exception:
                pass
        if hasattr(value, "__dict__") or hasattr(type(value), "__slots__"):
            instance_attrs = getattr(value, "__dict__", {}) or {}
            slot_attrs: dict[str, Any] = {}
            for cls in getattr(type(value), "__mro__", ()):
                for slot in getattr(cls, "__slots__", ()):
                    if not slot.startswith("_") and slot not in instance_attrs:
                        try:
                            slot_attrs[slot] = getattr(value, slot)
                        except AttributeError:
                            pass
            serialized: dict[str, Any] = {}
            for k, v in list({**instance_attrs, **slot_attrs}.items()):
                if not k.startswith("_"):
                    try:
                        serialized[k] = UnstructuredParser._serialize_metadata_value(v)
                    except (TypeError, ValueError):
                        serialized[k] = str(v)
            return serialized
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        # Preserve the source extension so unstructured can detect the file type.
        suffix = os.path.splitext(blob.source)[1] if blob.source else ""
        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                temp_file.write(blob.as_bytes())

            loader = UnstructuredLoader(
                file_path=temp_path,
                metadata_filename=blob.source or temp_path,
                **self._partition_kwargs,
            )

            for doc in loader.lazy_load():
                doc_meta: dict[str, Any] = dict(doc.metadata or {})
                category = doc_meta.get("category") or doc_meta.get("type") or "Uncategorized"
                text_as_html = doc_meta.get("text_as_html")
                raw_text = doc.page_content

                # Content selection strategy (optimized for CLIP / Jina AI):
                # we decide the string representation BEFORE building metadata.
                if category == "Formula":
                    page_content = (
                        text_as_html
                        or doc_meta.get("formula")
                        or raw_text
                        or "[Formula]"
                    )
                elif category == "Table":
                    # HTML table structures play well with multimodal embedders.
                    page_content = text_as_html or raw_text or "[Table]"
                elif category == "Image":
                    page_content = raw_text or f"Visual element: {category}"
                else:
                    page_content = raw_text or f"[{category}]"

                # Skip "ghost" elements (no text, no HTML, no image data).
                has_image = "image_base64" in doc_meta
                if (page_content is None or not str(page_content).strip()) and not has_image:
                    continue

                # Build enhanced metadata: start from the blob's, then layer the
                # element metadata on top.
                metadata: dict[str, Any] = blob.metadata.copy() if blob.metadata else {}
                metadata.update(doc_meta)
                metadata.update({
                    "element_type": category,
                    "has_formula": category == "Formula",
                })

                # Capture rich, category-specific data.
                if category == "Formula":
                    metadata["formula_latex"] = page_content
                elif category == "Table" and text_as_html:
                    metadata["table_html"] = text_as_html
                elif category == "Image":
                    if has_image:
                        metadata["image_data"] = doc_meta["image_base64"]
                    if "image_path" in doc_meta:
                        metadata["image_path"] = doc_meta["image_path"]

                # Coordinates are already serialized by `element.to_dict()`,
                # but we run through the safety net just in case.
                coords = doc_meta.get("coordinates")
                if coords is not None:
                    metadata["coordinates"] = self._serialize_metadata_value(coords)

                page_num = doc_meta.get("page_number")
                if page_num is not None:
                    try:
                        metadata["page_number"] = int(page_num)
                    except (TypeError, ValueError):
                        metadata["page_number"] = page_num

                # Final defensive serialization for DB compatibility.
                metadata = self._serialize_metadata_value(metadata)

                yield Document(page_content=str(page_content), metadata=metadata)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
