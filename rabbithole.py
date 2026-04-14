from typing import Dict
import nltk
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredImageLoader,
    UnstructuredPDFLoader,
)
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from cat import hook, BillTheLizard, EmbedderSettings
from cat.core_plugins.base_plugin.parsers import TableParser
from cat.services.service_factory import ServiceFactory

from .parsers import PowerPointParser, UnstructuredParser, YoutubeParser


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


@hook(priority=1)
async def rabbithole_instantiates_parsers(file_handlers: Dict, cat) -> Dict:
    lizard = BillTheLizard()
    sp = ServiceFactory(
        agent_key=lizard.agent_key,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )
    embedder_config: EmbedderSettings | None = await sp.get_config_class_from_adapter(await lizard.embedder())
    if not embedder_config:
        return file_handlers

    word_parser = (
        MsWordParser()
        if not embedder_config.is_multimodal()
        else UnstructuredParser(UnstructuredWordDocumentLoader)
    )
    powerpoint_parser = (
        PowerPointParser()
        if not embedder_config.is_multimodal()
        else UnstructuredParser(UnstructuredPowerPointLoader)
    )
    excel_parser = (
        TableParser()
        if not embedder_config.is_multimodal()
        else UnstructuredParser(UnstructuredExcelLoader)
    )

    file_handlers.update({
        "application/msword": word_parser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": word_parser,
        "application/vnd.ms-powerpoint": powerpoint_parser,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": powerpoint_parser,
        "application/pdf": (
            UnstructuredParser(UnstructuredPDFLoader)
            if embedder_config.is_multimodal()
            else file_handlers["application/pdf"]
        ),
        "application/vnd.ms-excel": excel_parser,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": excel_parser,
        "video/mp4": YoutubeParser(),
        "audio/mpeg": FasterWhisperParser(),
        "audio/mp3": FasterWhisperParser(),
        "audio/ogg": FasterWhisperParser(),
        "audio/wav": FasterWhisperParser(),
        "audio/webm": FasterWhisperParser(),
    })

    if embedder_config.is_multimodal():
        file_handlers.update({
            "image/png": UnstructuredParser(UnstructuredImageLoader),
            "image/jpeg": UnstructuredParser(UnstructuredImageLoader),
            "image/jpg": UnstructuredParser(UnstructuredImageLoader),
            "image/gif": UnstructuredParser(UnstructuredImageLoader),
            "image/bmp": UnstructuredParser(UnstructuredImageLoader),
            "image/tiff": UnstructuredParser(UnstructuredImageLoader),
            "image/webp": UnstructuredParser(UnstructuredImageLoader),
        })

    return file_handlers
