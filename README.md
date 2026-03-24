# Grinning Cat Plus

`Grinning Cat Plus` is a plugin for the Grinning Cat / Cheshire Cat ecosystem that extends the default factory with additional:

- LLM providers
- embedders (including multimodal embedders)
- chunkers
- file managers
- parsers for richer ingestion workflows

The plugin also adds multimodal parsing behavior (notably image handling) when the active embedder supports multimodality.

## Features

- Registers extra providers through factory hooks in `factories.py`
- Registers additional rabbit-hole parsers through `rabbithole.py`
- Supports cloud and local storage backends
- Supports semantic, token, hierarchical, and math-aware chunking strategies
- Enables image-aware document ingestion when using multimodal embedders

## Project layout

- `plugin.json`: plugin metadata and dependency declaration (`base_plugin`)
- `requirements.txt`: Python dependencies for all integrations
- `factories.py`: hook registration for allowed LLMs, embedders, chunkers, and file managers
- `rabbithole.py`: parser wiring and multimodal parser switching
- `llm/`: LLM config models and custom adapters
- `embedder/`: embedder config models and custom adapters
- `chunker/`: chunker config models and implementations
- `file_manager/`: storage/file manager config models and implementations
- `parsers/`: parser implementations for PowerPoint, unstructured content, and YouTube

## Registered providers

### LLMs

Enabled via `factory_allowed_llms` in `factories.py`:

- Cohere
- OpenAI Chat
- OpenAI Completions
- OpenAI-compatible APIs
- Ollama
- Google Gemini
- Azure OpenAI (chat + completion)
- Hugging Face Endpoint
- Hugging Face Text Generation Inference
- Anthropic
- Mistral AI
- Groq

### Embedders

Enabled via `factory_allowed_embedders` in `factories.py`:

- Qdrant FastEmbed (local)
- OpenAI
- Azure OpenAI
- Cohere embeddings
- Gemini embeddings
- OpenAI-compatible embeddings
- Fake/default embedder
- Mistral embeddings
- Ollama embeddings
- Jina embeddings
- Qwen3 local embeddings
- Qwen3 via Ollama
- Qwen3 via DeepInfra (OpenAI-compatible)
- Qwen3 via Text Embeddings Inference (TEI)
- Jina multimodal embedder
- Jina CLIP multimodal embedder
- VoyageAI embeddings

### File managers

Enabled via `factory_allowed_file_managers` in `factories.py`:

- AWS S3
- Azure Blob Storage
- Google Cloud Storage
- DigitalOcean Spaces

### Chunkers

Enabled via `factory_allowed_chunkers` in `factories.py`:

- Semantic chunker
- HTML semantic chunker
- JSON chunker
- spaCy token chunker
- NLTK token chunker
- Hierarchical chunker
- Math-aware hierarchical chunker

## Parser behavior and supported content types

The parser hook `rabbithole_instantiates_parsers` dynamically switches behavior depending on whether the active embedder is multimodal.

### Always wired

- Word (`application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
- PowerPoint (`application/vnd.ms-powerpoint`, `application/vnd.openxmlformats-officedocument.presentationml.presentation`)
- Excel (`application/vnd.ms-excel`, `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`)
- Video (`video/mp4`) via YouTube transcript extraction
- Audio (`audio/mpeg`, `audio/mp3`, `audio/ogg`, `audio/wav`, `audio/webm`) via `FasterWhisperParser`

### Conditional behavior

- PDF (`application/pdf`): uses `UnstructuredParser` when embedder is multimodal; otherwise falls back to the existing handler
- Images (`image/png`, `image/jpeg`, `image/jpg`, `image/gif`, `image/bmp`, `image/tiff`, `image/webp`): only added when embedder is multimodal

### Notable parser details

- `parsers/unstructured_parser.py` enriches metadata with element type, table HTML, formula data, coordinates, page number, and optional image payload
- `parsers/youtube_parser.py` fetches transcript text for YouTube sources (languages set to `en` and `it`)
- `rabbithole.py` downloads NLTK assets (`punkt`, `averaged_perceptron_tagger`) at import time

## Installation

Because this is a plugin, installation depends on your Grinning Cat/Cheshire Cat deployment strategy.

Typical local flow:

```bash
# from your plugin root
pip install -r requirements.txt
```

Then install/enable the plugin in your host application and select providers from the admin settings UI.

## Configuration

Each provider is configured through its corresponding Pydantic settings class under:

- `llm/configs.py`
- `embedder/configs.py`
- `chunker/configs.py`
- `file_manager/configs.py`

In the admin UI, these appear using each class `humanReadableName` and expose the required fields (API keys, endpoints, model names, chunking params, etc.).

## Compatibility notes

- Some integrations are intentionally commented out due to compatibility concerns (see `requirements.txt` and config files)
- Multimodal parsing paths rely on `unstructured` extras and related dependencies
- Cloud providers require valid credentials and service-specific setup

## Development

Useful files while extending the plugin:

- Add/remove available providers: `factories.py`
- Change ingestion/parser routing: `rabbithole.py`
- Add a new provider config: `*/configs.py`
- Implement provider logic: `*/custom.py`
- Add parser modules: `parsers/`

## Metadata

From `plugin.json`:

- Name: `Grinning Cat Plus`
- Version: `0.0.1`
- Author: `Matteo Cacciola`
- URL: `https://github.com/matteocacciola/grinning_cat_plus`

