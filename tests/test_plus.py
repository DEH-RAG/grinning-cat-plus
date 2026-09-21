"""Tests for the ``grinning_cat_plus`` core plugin on upstream CAT.

Verifies the plugin loads on upstream CAT and its hooks work:
- the plugin is discovered and active on the system agent (lizard);
- the four ``factory_allowed_*`` hooks extend the allowed provider lists with
  the plugin's config classes (LLMs incl. the dynamic OpenRouter config,
  embedders, chunkers, file managers);
- ``rabbithole_instantiates_parsers`` builds the parser map (word / ppt /
  excel / odf / pdf / audio / video) against the active text-only embedder;
- the plugin's own ``after_cheshire_cat_creation`` flow hook fires on agent
  creation (it toggles the plugin on the fresh agent's plugin manager).

Call-path notes (mirror production):
- the LLM / chunker / file-manager factories run on the AGENT's plugin manager
  (``_bot_factory_services_params``), so those hooks are executed there;
- the embedder factory runs ONLY on the system agent's plugin manager
  (``_orchestrator_factory_services_params``), where ``context_execute_hook``
  is ``"lizard"`` — matching the ``factory_allowed_embedders(allowed, lizard)``
  signature used by both this plugin and upstream ``base_plugin``.

Uses ``tests/conftest.py`` fixtures: Redis db=1 (isolated).
"""

import pytest

from cat.plugins.grinning_cat_plus import factories, flow, rabbithole


@pytest.fixture
async def plus_plugin(cheshire_cat):
    """Return the cheshire cat with the plugin guaranteed active on its manager.

    The plugin's own ``after_cheshire_cat_creation`` hook toggles the plugin on
    a freshly created agent. As a core plugin the agent already has it active,
    so the toggle deactivates it; re-activate so agent-level hook execution
    works.
    """
    pm = cheshire_cat.plugin_manager
    if "grinning_cat_plus" not in pm.active_plugins:
        await pm.activate_plugin("grinning_cat_plus")
    return cheshire_cat


async def test_plugin_is_active_core_plugin(cheshire_cat):
    """The plugin is discovered and active on the system agent (lizard)."""
    pm = cheshire_cat.lizard.plugin_manager
    assert "grinning_cat_plus" in pm.active_plugins
    assert "grinning_cat_plus" in pm.plugins
    assert "factory_allowed_llms" in pm.hooks


async def test_factory_allowed_llms_hook(plus_plugin):
    allowed = await plus_plugin.plugin_manager.execute_hook(
        "factory_allowed_llms", [], caller=plus_plugin
    )
    names = {c.__name__ for c in allowed}
    assert "LLMOpenAIChatConfig" in names
    assert "LLMOpenRouterConfig" in names  # dynamic subclass keeps the same __name__


async def test_factory_allowed_embedders_hook(plus_plugin):
    """Embedder factory runs on the system agent's plugin manager (production path)."""
    pm = plus_plugin.lizard.plugin_manager
    allowed = await pm.execute_hook("factory_allowed_embedders", [], caller=plus_plugin.lizard)
    names = {c.__name__ for c in allowed}
    assert "EmbedderQdrantFastEmbedConfig" in names
    assert "VllmMultimodalConfiguration" in names


async def test_factory_allowed_chunkers_hook(plus_plugin):
    allowed = await plus_plugin.plugin_manager.execute_hook(
        "factory_allowed_chunkers", [], caller=plus_plugin
    )
    names = {c.__name__ for c in allowed}
    assert "SemanticChunkerSettings" in names
    assert "MathAwareSemanticChunkerSettings" in names


async def test_factory_allowed_file_managers_hook(plus_plugin):
    allowed = await plus_plugin.plugin_manager.execute_hook(
        "factory_allowed_file_managers", [], caller=plus_plugin
    )
    names = {c.__name__ for c in allowed}
    assert "AWSFileManagerConfig" in names


async def test_rabbithole_instantiates_parsers_text_only(plus_plugin):
    """Parser map is built for the active (default text-only) embedder."""
    handlers = await plus_plugin.plugin_manager.execute_hook(
        "rabbithole_instantiates_parsers", {}, caller=plus_plugin
    )
    # word / ppt / excel / odf
    assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in handlers
    assert "application/vnd.openxmlformats-officedocument.presentationml.presentation" in handlers
    assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in handlers
    assert "application/vnd.oasis.opendocument.text" in handlers
    # pdf handled through the unstructured fast path
    assert "application/pdf" in handlers
    # audio / video
    assert "video/mp4" in handlers
    assert "audio/mpeg" in handlers
    # text-only embedder: no image handlers
    assert "image/png" not in handlers


async def test_after_cheshire_cat_creation_toggles_plugin(cheshire_cat):
    """The flow hook is registered and fired on agent creation.

    It toggles the plugin on the fresh agent's plugin manager; as a core plugin
    the agent already had it active, so the toggle deactivated it — proving the
    hook ran during ``create_cheshire_cat``.
    """
    pm = cheshire_cat.lizard.plugin_manager
    hook_ids = {h.plugin_id for h in pm.hooks["after_cheshire_cat_creation"]}
    assert "grinning_cat_plus" in hook_ids
    # the toggle ran on the fresh agent's plugin manager
    assert "grinning_cat_plus" not in cheshire_cat.plugin_manager.active_plugins