"""The recommended presets: the production settings, runnable on one machine."""

from __future__ import annotations

import pytest
import yaml

from virtual_context.config import load_config, validate_config
from virtual_context.presets import get_preset

PRESETS = ["recommended", "recommended-local"]


def _config(name):
    preset = get_preset(name)
    assert preset is not None
    return preset, load_config(config_dict=yaml.safe_load(preset.template))


@pytest.mark.parametrize("name", PRESETS)
def test_the_written_file_is_the_preset(name):
    preset, _ = _config(name)
    assert yaml.safe_load(preset.template) == preset.config_dict


@pytest.mark.parametrize("name", PRESETS)
def test_it_validates(name):
    _, config = _config(name)
    assert validate_config(config) == []


@pytest.mark.parametrize("name", PRESETS)
def test_models_do_the_tagging_and_summaries_and_the_tools_are_on(name):
    _, config = _config(name)
    assert config.tag_generator.type == "llm" and config.tag_generator.model
    assert config.summarization.provider and config.summarization.model
    assert config.paging.enabled
    assert config.retriever.inbound_tagger_type == "embedding"


@pytest.mark.parametrize("name", PRESETS)
def test_it_needs_no_database_server_or_judgment_key(name):
    _, config = _config(name)
    assert config.storage.backend == "sqlite"
    assert config.retriever.vector_search_enabled is False
    assert config.judgment.mode == "legacy" and not config.judgment.seams


def test_the_two_presets_differ_only_in_where_the_model_runs():
    hosted, local = (yaml.safe_load(get_preset(n).template) for n in PRESETS)
    for section in ("tag_generator", "summarization"):
        for key in ("provider", "model"):
            hosted[section].pop(key)
            local[section].pop(key)
    hosted.pop("providers"), local.pop("providers")
    assert hosted == local


def test_next_steps_name_what_each_needs():
    assert any("OPENROUTER_API_KEY" in step for step in get_preset("recommended").next_steps)
    assert any("model server" in step for step in get_preset("recommended-local").next_steps)
