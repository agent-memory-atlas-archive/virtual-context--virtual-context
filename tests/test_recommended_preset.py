"""The recommended preset: the production settings, runnable on one machine."""

from __future__ import annotations

import yaml

from virtual_context.config import load_config, validate_config
from virtual_context.presets import get_preset


def _config():
    preset = get_preset("recommended")
    assert preset is not None
    return preset, load_config(config_dict=yaml.safe_load(preset.template))


def test_the_written_file_is_the_preset():
    preset, _ = _config()
    assert yaml.safe_load(preset.template) == preset.config_dict


def test_it_validates():
    _, config = _config()
    assert validate_config(config) == []


def test_models_do_the_tagging_and_summaries_and_the_tools_are_on():
    _, config = _config()
    assert config.tag_generator.type == "llm" and config.tag_generator.model
    assert config.summarization.provider and config.summarization.model
    assert config.paging.enabled
    assert config.retriever.inbound_tagger_type == "embedding"


def test_it_needs_no_database_server_or_judgment_key():
    _, config = _config()
    assert config.storage.backend == "sqlite"
    assert config.retriever.vector_search_enabled is False
    assert config.judgment.mode == "legacy" and not config.judgment.seams


def test_its_next_steps_name_the_key_it_needs():
    preset, _ = _config()
    assert any("OPENROUTER_API_KEY" in step for step in preset.next_steps)
