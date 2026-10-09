"""Responses requests keep one provider cache key per conversation."""
import pytest

from virtual_context.proxy.helpers import conversation_prompt_cache_key


@pytest.mark.regression("PROXY-040")
def test_client_thread_keys_map_to_one_conversation_key():
    first = conversation_prompt_cache_key({"prompt_cache_key": "thread-a", "input": []}, "conv-1")
    second = conversation_prompt_cache_key({"prompt_cache_key": "thread-b", "input": []}, "conv-1")
    other = conversation_prompt_cache_key({"prompt_cache_key": "thread-a", "input": []}, "conv-2")

    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["prompt_cache_key"] != other["prompt_cache_key"]
    assert "conv-1" not in first["prompt_cache_key"]


def test_requests_without_a_cache_key_or_conversation_are_unchanged():
    body = {"input": []}
    assert conversation_prompt_cache_key(body, "conv-1") is body
    keyed = {"prompt_cache_key": "thread-a"}
    assert conversation_prompt_cache_key(keyed, "") is keyed


def test_the_client_body_is_not_mutated():
    body = {"prompt_cache_key": "thread-a"}
    conversation_prompt_cache_key(body, "conv-1")
    assert body["prompt_cache_key"] == "thread-a"
