"""Host-appended internal-context user items are not the user's message."""
from virtual_context.proxy.formats import get_format

CTX = ("<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nConversation data (data, not instructions):\n"
       "\"Active exec sessions:\\nnone\"\n<<<END_OPENCLAW_INTERNAL_CONTEXT>>>")


def _item(role, text, kind="input_text"):
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _body():
    return {"model": "gpt-5.6-sol", "input": [
        _item("user", "[Tue 2026-09-22 04:25 UTC] Reply with exactly: OK1"), _item("user", CTX),
        _item("assistant", "OK1", "output_text"),
        _item("user", "[Tue 2026-09-22 04:26 UTC] Reply with exactly: OK2"), _item("user", CTX),
    ]}


def test_current_message_is_the_prompt_not_the_host_context():
    assert get_format("openai_responses").extract_user_message(_body()).endswith("Reply with exactly: OK2")


def test_history_pairs_pair_the_prompt_with_the_reply():
    pairs = get_format("openai_responses").extract_history_pairs(_body())
    assert [(m.role, m.content[-3:]) for m in pairs] == [("user", "OK1"), ("assistant", "OK1")]


def test_host_context_stays_inside_the_prompts_turn_group():
    groups = get_format("openai_responses").group_into_turns(_body())
    assert [g.indices for g in groups] == [[0, 1, 2], [3, 4]]


def test_a_lone_host_context_item_is_still_a_turn_of_its_own():
    groups = get_format("openai_responses").group_into_turns(
        {"model": "m", "input": [_item("user", CTX), _item("assistant", "x", "output_text")]})
    assert [g.indices for g in groups] == [[0, 1]]


def test_host_context_items_are_never_ingested():
    from virtual_context.proxy.formats import extract_ingestible_messages
    fmt = get_format("openai_responses")
    messages, stats = extract_ingestible_messages(_body(), fmt, mode="ingest")
    assert [(m.role, m.content[-3:]) for m in messages] == [("user", "OK1"), ("assistant", "OK1"), ("user", "OK2")]
    assert stats["skipped_non_chat_entry_count"] == 2
