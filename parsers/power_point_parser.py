import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document


# MIME types that are ZIP-based and confuse libmagic → we convert via LibreOffice
_ODP_MIME = "application/vnd.oasis.opendocument.presentation"

_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_PPT_MIME  = "application/vnd.ms-powerpoint"

# content_type hints for partition() — avoids libmagic misdetection
_MIME_HINTS = {
    ".pptx": _PPTX_MIME,
    ".ppt":  _PPT_MIME,
    ".odp":  _ODP_MIME,
}


def _libreoffice_bin() -> str | None:
    return shutil.which("libreoffice") or shutil.which("soffice")


def _libreoffice_convert(source_path: str, target_ext: str) -> str:
    binary = _libreoffice_bin()
    if not binary:
        raise RuntimeError(
            "LibreOffice/soffice non trovato nel PATH: "
            "necessario per convertire file ODF"
        )
    src = Path(source_path)
    outdir = Path(tempfile.mkdtemp(prefix="odf-convert-"))
    try:
        proc = subprocess.run(
            [binary, "--headless", "--convert-to", target_ext,
             "--outdir", str(outdir), str(src)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Conversione LibreOffice fallita ({proc.returncode}): "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        converted = outdir / f"{src.stem}.{target_ext}"
        if not converted.exists():
            files = ", ".join(p.name for p in outdir.iterdir())
            raise RuntimeError(
                f"File convertito non trovato: atteso {converted.name}, "
                f"presenti: {files}"
            )
        return str(converted)
    except Exception:
        shutil.rmtree(outdir, ignore_errors=True)
        raise


class PowerPointParser(BaseBlobParser):
    """Lightweight PowerPoint/ODP parser used in non-multimodal mode.

    Strategy:
    - .pptx / .ppt  → partition() with content_type forced to avoid libmagic UNK
    - .odp          → LibreOffice headless converts to .pptx, then partition_pptx()

    NOTE: Blob.as_temp_file() is not available in all langchain-core versions;
    we manage the temp file manually via tempfile.NamedTemporaryFile.
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        source = blob.source or ""
        suffix = os.path.splitext(source)[1].lower()
        temp_path: str | None = None
        converted_path: str | None = None
        converted_dir: str | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                tmp.write(blob.as_bytes())

            if suffix == ".odp":
                # Convert ODP → PPTX via LibreOffice, then use partition_pptx directly
                from unstructured.partition.pptx import partition_pptx
                converted_path = _libreoffice_convert(temp_path, "pptx")
                converted_dir = str(Path(converted_path).parent)
                elements = partition_pptx(filename=converted_path)
            else:
                # Force content_type so partition() skips libmagic detection
                from unstructured.partition.auto import partition
                content_type = _MIME_HINTS.get(suffix)
                kwargs = {}
                if content_type:
                    kwargs["content_type"] = content_type
                elements = partition(filename=temp_path, **kwargs)

            for element in elements:
                text = getattr(element, "text", None) or ""
                if not text.strip():
                    continue
                metadata = blob.metadata.copy() if blob.metadata else {}
                metadata.update({
                    "source": source,
                    "element_type": getattr(element, "category", "Unknown"),
                    "page_number": getattr(
                        getattr(element, "metadata", None), "page_number", None
                    ),
                })
                yield Document(page_content=text, metadata=metadata)

        finally:
            for path in (temp_path, converted_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
            if converted_dir and os.path.isdir(converted_dir):
                shutil.rmtree(converted_dir, ignore_errors=True)
