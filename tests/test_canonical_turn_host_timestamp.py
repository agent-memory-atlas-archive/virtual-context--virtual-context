"""The host's send-time stamp on a user message does not change its canonical identity."""
from virtual_context.core.canonical_turns import compute_turn_hash_from_raw, normalize_turn_text


def test_leading_host_timestamp_is_not_part_of_the_normalized_text():
    assert normalize_turn_text("[Tue 2026-09-22 04:53 UTC] Reply with exactly: OK1") == "Reply with exactly: OK1"
    assert normalize_turn_text("  [Mon 2026-01-05 23:59 UTC]\nhello") == "hello"


def test_stamped_and_unstamped_copies_hash_the_same():
    stamped = compute_turn_hash_from_raw("[Tue 2026-09-22 04:53 UTC] Reply with exactly: OK1", "OK1")
    plain = compute_turn_hash_from_raw("Reply with exactly: OK1", "OK1")
    assert stamped == plain


def test_only_a_leading_stamp_is_removed():
    text = "see [Tue 2026-09-22 04:53 UTC] in the log"
    assert normalize_turn_text(text) == text
    assert normalize_turn_text("[Tue 2026-09-22 04:53] no zone") == "[Tue 2026-09-22 04:53] no zone"


def test_a_message_that_itself_begins_with_a_stamp_normalizes_the_same_from_either_copy():
    typed = "[Mon 2026-09-21 04:53 UTC] hello"
    assert normalize_turn_text(typed) == normalize_turn_text("[Tue 2026-09-22 04:53 UTC] " + typed) == "hello"
