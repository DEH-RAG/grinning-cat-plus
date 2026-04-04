from typing import List
from cat import hook, BillTheLizard
from cat.db.cruds import settings as crud_settings
from cat.services.factory.chunker import ChunkerSettings
from cat.services.factory.embedder import EmbedderSettings
from cat.services.factory.file_manager import FileManagerConfig
from cat.services.factory.llm import LLMSettings

from .chunkers.configs import (
    SemanticChunkerSettings,
    HTMLSemanticChunkerSettings,
    JSONChunkerSettings,
    TokenSpacyChunkerSettings,
    TokenNLTKChunkerSettings,
    HierarchicalChunkerSettings,
    MathAwareHierarchicalChunkerSettings,
)
from .embedders.configs import (
    EmbedderQdrantFastEmbedConfig,
    EmbedderOpenAIConfig,
    EmbedderAzureOpenAIConfig,
    EmbedderGeminiChatConfig,
    EmbedderOpenAICompatibleConfig,
    EmbedderCohereConfig,
    EmbedderMistralAIChatConfig,
    EmbedderVoyageAIChatConfig,
    EmbedderOllamaConfig,
    EmbedderJinaConfig,
    Qwen3LocalEmbeddingsConfig,
    Qwen3OllamaEmbeddingsConfig,
    Qwen3DeepInfraEmbeddingsConfig,
    Qwen3TEIEmbeddingsConfig,
    EmbedderJinaMultimodalConfig,
    JinaCLIPEmbeddingsConfig,
)
from .file_managers.configs import (
    AWSFileManagerConfig,
    AzureFileManagerConfig,
    GoogleFileManagerConfig,
    DigitalOceanFileManagerConfig,
)
from .llms.configs import (
    LLMOpenAIChatConfig,
    LLMOpenAIConfig,
    LLMOpenAICompatibleConfig,
    LLMOllamaConfig,
    LLMGeminiChatConfig,
    LLMCohereConfig,
    LLMAzureOpenAIConfig,
    LLMAzureChatOpenAIConfig,
    LLMHuggingFaceEndpointConfig,
    LLMHuggingFaceTextGenInferenceConfig,
    LLMAnthropicChatConfig,
    LLMMistralAIChatConfig,
    LLMGroqChatConfig,
)


@hook(priority=1)
def factory_allowed_llms(allowed: List[LLMSettings], cat) -> List:
    return allowed + [
        LLMOpenAIChatConfig,
        LLMOpenAIConfig,
        LLMOpenAICompatibleConfig,
        LLMOllamaConfig,
        LLMGeminiChatConfig,
        LLMCohereConfig,
        LLMAzureOpenAIConfig,
        LLMAzureChatOpenAIConfig,
        LLMHuggingFaceEndpointConfig,
        LLMHuggingFaceTextGenInferenceConfig,
        LLMAnthropicChatConfig,
        LLMMistralAIChatConfig,
        LLMGroqChatConfig,
    ]


@hook(priority=1)
def factory_allowed_embedders(allowed: List[EmbedderSettings], lizard) -> List:
    return allowed + [
        EmbedderQdrantFastEmbedConfig,
        EmbedderOpenAIConfig,
        EmbedderAzureOpenAIConfig,
        EmbedderGeminiChatConfig,
        EmbedderOpenAICompatibleConfig,
        EmbedderCohereConfig,
        EmbedderMistralAIChatConfig,
        EmbedderVoyageAIChatConfig,
        EmbedderOllamaConfig,
        EmbedderJinaConfig,
        Qwen3LocalEmbeddingsConfig,
        Qwen3OllamaEmbeddingsConfig,
        Qwen3DeepInfraEmbeddingsConfig,
        Qwen3TEIEmbeddingsConfig,
        EmbedderJinaMultimodalConfig,
        JinaCLIPEmbeddingsConfig,
    ]


@hook(priority=1)
def factory_allowed_file_managers(allowed: List[FileManagerConfig], cat) -> List:
    return allowed + [
        AWSFileManagerConfig,
        AzureFileManagerConfig,
        GoogleFileManagerConfig,
        DigitalOceanFileManagerConfig,
    ]


@hook(priority=1)
def factory_allowed_chunkers(allowed: List[ChunkerSettings], cat) -> List:
    return allowed + [
        SemanticChunkerSettings,
        HTMLSemanticChunkerSettings,
        JSONChunkerSettings,
        TokenSpacyChunkerSettings,
        TokenNLTKChunkerSettings,
        HierarchicalChunkerSettings,
        MathAwareHierarchicalChunkerSettings,
    ]


@hook(priority=1)
async def lizard_notify_plugin_installation(plugin_id: str, plugin_path: str, lizard: BillTheLizard):
    this_plugin_id = lizard.mad_hatter.get_plugin().id
    if this_plugin_id != plugin_id:
        return

    # for each Cheshire Cat, activate this plugin
    ccat_ids = await crud_settings.get_agents_main_keys()
    for ccat_id in ccat_ids:
        if (ccat := await lizard.get_cheshire_cat(ccat_id)) is None:
            continue

        ccat.plugin_manager.toggle_plugin(plugin_id)
