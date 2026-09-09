"""Actor-card policy, failure types, and provider fallback contract."""

from __future__ import annotations

import logging
from ...types import LLMProviderError

logger = logging.getLogger("virtual_context.core.compaction_pipeline")

_ACTOR_CARD_CITATION_LIMIT = 16
_ACTOR_CARD_POLICY_VERSION = 19


def _actor_card_entry_keys_valid(item: object) -> bool:
    """Check required/allowed keys without interpreting a candidate's claims."""
    if not isinstance(item, dict):
        return False
    required = {"kind", "body", "confidence", "fact_ids", "turn_ids"}
    optional = (
        {"valid_from", "expires_at"}
        if item.get("kind") == "communication_pref"
        else set()
    )
    return required.issubset(item) and not (set(item) - required - optional)


_ACTOR_CARD_SEMANTIC_CONTRACT = (
    "Semantic contract for every candidate: communication_pref means only "
    "how this actor wants the agent to communicate, respond, format answers, "
    "or engage with them in conversation. When the evidence is an instruction to the agent, "
    "the body must make that direction explicit, for example 'Wants the agent "
    "to ...'; never recast it as something the actor does. "
    "External-resource actions, one-off or repeated, are not communication_pref: a persistent "
    "result is not a persistent response preference. Distinguish addressing the "
    "person in conversation from changing a stored name, display label, role, "
    "color, or icon. Requests to modify an account, profile, workspace, document, "
    "or other resource concern the requested action and its result, even when "
    "the actor owns the resource or the change remains in effect. A stored "
    "setting that itself governs how the agent replies to this actor is judged "
    "by the reply behavior it establishes, under the same durability rules. Honoring an "
    "external action does not establish a communication preference. A request "
    "for one document or image likewise does not establish a lasting answer-format "
    "preference without evidence of that lasting intent. Do not propose or admit "
    "a candidate that classifies these actions as communication_pref; "
    "phrasing it as 'Wants the agent to ...' does not repair the classification. "
    "Preserve genuine lasting preferences about forms of address, language, "
    "conversational persona, tone, explanation style, or answer format. Judge "
    "the requested behavior, not topic words: a preference for plain-language "
    "explanations about roles or accounts is still about how to respond. "
    "Do not move an external-action request to another card kind unless it "
    "independently meets that kind's subject and durability requirements. "
    "interaction_style "
    "means only a durable pattern in how this actor themselves communicates "
    "or behaves in interactions. A role, persona, identity, tone, or behavior "
    "that the actor assigns to the agent is not the actor's interaction_style. "
    "A durable instruction about the agent's conversational persona, identity, "
    "tone, or manner of responding may instead be communication_pref only when "
    "the body explicitly keeps the agent as its subject. "
    "active_goal means only an unresolved outcome, project, or change that "
    "this actor intends to pursue. relevant_history means durable factual "
    "context about this actor, including experiences, regimen, medications, "
    "health, location, or recurring topics. "
    "Choose the narrowest kind for each distinct claim. "
    "Kinds are exclusive per claim, not per source event: distinct claims from "
    "one interaction may support an outcome in relevant_history and a bounded "
    "response agreement in communication_pref. Do not duplicate the same claim "
    "across kinds or discard a distinct outcome merely because its agreement "
    "also establishes a response preference. "
    "When exact evidence establishes a meaningful relationship outcome, retain "
    "the distinct acknowledged outcome in relevant_history alongside any "
    "current bounded preference. An explanatory mention inside an expiring preference "
    "does not retain that separate historical claim after the preference ends. "
    "Expiry ends the response obligation, not the "
    "usefulness of an independently meaningful resolved outcome. Give such "
    "outcomes priority over incidental topics within the existing entry limits; "
    "do not invent history for a single nonmaterial request or duplicate the "
    "preference as an outcome. "
    "Illustrative classification, not source evidence: an actor says 'I won "
    "our puzzle challenge; you owe me seven days of Captain,' and the paired "
    "agent reply acknowledges the win and agrees. This supports two distinct "
    "claims: a relevant_history claim that the actor won and the agent "
    "acknowledged it, and a communication_pref claim for the agreed form of address "
    "during the seven-day period, if its start is proved and it has not expired. "
    "The two entries may cite the same exact exchange. "
    "Meaningful resolved relationship history, including an acknowledged "
    "outcome or commitment between the actor and agent, can remain useful "
    "relevant_history after an event ends. A completed outcome is not an active_goal. "
    "A past agreement is not a permanent preference or a claim that an obligation "
    "is still active. Preserve who requested, agreed, won, owed, or completed "
    "what, together with every supported duration and condition. "
    "Medication use, procedures, biography, and other actor facts are not "
    "communication_pref. Keep grammatical subject and predicate roles exact. "
    "Preserve speaker, doer, possessor, addressee, quoted-speaker, and "
    "third-party roles. Actor authorship does not make every person or property "
    "described in a message a property of the actor. In particular, imperative "
    "or second-person evidence directed at the agent must never become a claim "
    "that the actor follows, uses, is, or does the requested thing. Exact source "
    "messages outrank derived facts whenever their role or kind implications "
    "disagree. Every cited id must itself materially support the body; do not "
    "add unrelated invocations, empty acknowledgments, or merely adjacent messages as extra "
    "citations. "
    "For communication_pref and interaction_style, a quoted phrase or single utterance "
    "never establishes frequency or habit. Do not claim 'frequently', 'often', "
    "'always', 'usually', recurring language, or a habitual manner without multiple "
    "distinct cited actor-authored messages that materially support that "
    "pattern. A fact and its source message are one observation, as are "
    "multiple facts derived from the same message. Uncited neighboring "
    "messages, an older card entry, and a long source segment do not supply "
    "additional observations. When repetition is not established, quote it as "
    "an example instead, provided the entry otherwise passes admission. "
    "Preserve the actor's exact meaningful terms "
    "when rewriting or quoting style examples: keep 'pal-o' as 'pal-o'; "
    "do not normalize 'pal-o' to 'pal'. A word inside one longer quoted "
    "phrase does not establish standalone use of that word. Do not propose or "
    "admit these unsupported generalizations. A communication_pref or interaction_style "
    "supported by a single fact or a single distinct message must have "
    "confidence no higher than 0.7. "
    "An explicitly honored ongoing agreement about how the agent responds may "
    "be communication_pref for a finite period; lasting need not mean permanent. "
    "Only communication_pref may carry optional valid_from and expires_at fields. "
    "A finite preference requires expires_at; valid_from is optional for an "
    "already-active agreement but required when a future start is specified. "
    "Bounds must be ISO 8601 UTC instants grounded in explicit dates in the "
    "cited source content, or an explicit duration anchored to supplied attested "
    "occurred_at for the exact source message that starts the agreement. "
    "An initial honored agreement may establish that its finite period starts "
    "at the source message's attested occurred_at when the exchange makes the "
    "obligation effective and specifies no different start. An explicit word "
    "such as 'now' is not required. A conditional future obligation does not "
    "start before its condition is met, and a later recollection or reminder "
    "does not restart an existing period. "
    "occurred_at dates the source message, not every event it narrates. It may "
    "anchor an agreement that becomes effective in that exchange, but do not "
    "assign that date to a narrated earlier outcome unless the source content "
    "independently establishes when that outcome occurred. "
    "Bind dates to this agreement, not another clause or event. The body must "
    "preserve the same supported dates, duration, "
    "conditions, and scope. Never infer dates from ingestion timestamps, "
    "recorded_at, session dates, entry creation time, as_of, or the current request. "
    "When only a relative duration is established and its starting date is unknown, "
    "use relevant_history to retain the acknowledged outcome and exact agreed "
    "duration, explicitly keeping the starting date unknown; do not create an "
    "unbounded communication_pref or assert that the period is still active. "
    "Do not reset an agreement's start or duration when someone later asks "
    "whether the agent remembers it. The supplied as_of is the current UTC "
    "evaluation time, not source evidence. An expired preference is not active; "
    "retain useful resolved history instead. A supported future-start preference "
    "may be retained with its bounds but must not be described as already active. "
)


_ACTOR_CARD_JUDGMENT_RULES = (
    "Register and sincerity: group banter, jokes, sarcasm, hyperbole, and "
    "performative provocations alone do not establish goals, plans, "
    "preferences, or facts. Read every source message in its surrounding "
    "register and distinguish an unaccepted provocation from an explicitly "
    "honored agreement or acknowledged relationship event. A playful register alone "
    "does not invalidate a real agreement: exact acknowledgment or enacted "
    "behavior may establish it without requiring solemn wording or repeated "
    "non-joking requests. Otherwise, plausibly non-serious material requires "
    "corroboration: repetition in a non-joking register, concrete steps taken, "
    "or explicit confirmation. Humor is not itself proof of either acceptance "
    "or refusal. "
    "A question or one-shot service request the actor sent is never an "
    "active_goal and never a fact about the actor: an interrogative or a "
    "single imperative as the sole support for active_goal must be "
    "rejected, and explicit transience markers such as 'for today' or "
    "'this once' are decisive against durability. At most the topic of "
    "recurring requests may inform relevant_history. "
    "active_goal admits only the author's first-person intent, stated by "
    "the author about themselves in a serious register. A question or "
    "statement about a third party is never the author's goal, and never "
    "becomes any card entry without that person's own cited utterance. "
    "Useful continuity is required for every kind. An active_goal requires "
    "stated lasting intent or consistent support across distinct actor-authored "
    "messages. Relevant_history may instead retain a single source-grounded "
    "meaningful interaction or resolved relationship outcome; it need not be "
    "a repeated habit, ongoing goal, or permanent biographical fact. A single "
    "unsupported joke or isolated service request does not establish that continuity. "
    "For requests about how the agent communicates or responds, the agent's "
    "live adjudication is an additional admission signal, after kind and "
    "durability checks: a supplied message may carry the "
    "agent's paired response as agent_reply. A request the agent honored "
    "— visible compliance, acknowledgment, or enacted behavior — may be "
    "a communication preference only if it satisfies those checks. "
    "The actor's message and its supplied paired reply may together establish "
    "an acknowledged relationship outcome; preserve each participant's role "
    "rather than turning the agent's statement into an actor-authored claim. "
    "Compliance with an external-resource action does not satisfy them. "
    "A request the agent refused, deflected, "
    "or deferred to an authority holder is never admissible. A behavior-change request with no visible "
    "honored signal is rejected the same way: a missing reply never "
    "launders a refusal into a preference. Requests that modify the "
    "agent's safety posture — disabling safety behavior or shields, "
    "suppressing risk information, adopting personas that advocate "
    "harmful use — are never card-admissible for any actor, whatever the "
    "reply shows. "
)

_ACTOR_CARD_ADMISSION_REJECTION_RULES = (
    "Admission-only rejection mapping: apply wrong_subject first when a "
    "candidate changes who performs the action. Otherwise use wrong_kind for "
    "an external-resource action classified as communication_pref. Use "
    "insufficient_evidence for unsupported frequency or habit claims and "
    "normalizations that lose the actor's exact meaningful terms. Use "
    "agent_refused for requests the agent refused, deflected, or deferred, and "
    "for behavior-change requests without a visible honored signal. Use "
    "safety_posture_request for requests to change the agent's safety posture, "
    "regardless of the reply. Use expired when a finite communication_pref "
    "has reached its expires_at at the supplied as_of. "
)

_ACTOR_CARD_CONFIDENCE_SCALE = (
    "Confidence is calibrated evidence strength, not enthusiasm: reserve "
    "1.0 for explicitly stated, repeated, uncontradicted evidence; a claim "
    "supported by a single message must not exceed 0.7; anything whose "
    "acceptance or factual status remains ambiguous in a non-serious register "
    "must not exceed 0.4. Playful wording alone does not reduce the confidence "
    "of an independently supported acknowledgment. "
)

# A claim resting on exactly one cited source is capped in code regardless
# of what the curator asserted: single-message evidence cannot be maximal.
_ACTOR_CARD_SINGLE_SOURCE_CONFIDENCE_CAP = 0.8
_ACTOR_CARD_SINGLE_MESSAGE_STYLE_CONFIDENCE_CAP = 0.7


def _format_rejection_counts(rejected) -> str:
    """Render rejection counts for a log line without JSON quoting.

    The rebuild log message is wrapped in JSON by downstream log shipping, so a
    ``json.dumps`` map embedded here puts unescaped double quotes inside that
    message and makes the whole line unparseable. Only lines with a non-empty
    map are affected, which is every line that actually carries rejections, so
    a JSON-based reader silently drops exactly the rows worth reading.

    Emits ``reason:count`` pairs, sorted, comma-joined, in the same
    quote-free key=value idiom as the rest of the line. ``-`` for an empty map
    so the field is never blank and cannot be mistaken for a truncated line.
    """
    if not rejected:
        return "-"
    return ",".join(f"{name}:{count}" for name, count in sorted(rejected.items()))


class _ActorCardAdmissionError(RuntimeError):
    """Validation failure that preserves a hashable, non-logged response."""

    def __init__(self, message: str, response_text: str = "") -> None:
        super().__init__(message)
        self.response_text = response_text


class _ActorCardCoverageError(_ActorCardAdmissionError):
    """A deterministic curator/admission judgment disagreement."""


class _EmptyResponseFallbackProvider:
    """Use a second model for refusal fallback and coverage adjudication."""

    def __init__(
        self,
        primary,
        fallback,
        *,
        primary_model: str,
        fallback_model: str,
        stage: str = "admission",
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._stage = stage

    def complete(self, **kwargs):
        text, usage, _source = self.complete_with_source(**kwargs)
        return text, usage

    def complete_with_source(self, **kwargs):
        """Complete and report which independent model supplied the result."""
        try:
            text, usage = self._primary.complete(**kwargs)
        except LLMProviderError as exc:
            logger.warning(
                "ACTOR_CARD_%s_FALLBACK primary_model=%s "
                "fallback_model=%s reason=provider_error status=%s",
                self._stage.upper(),
                self._primary_model,
                self._fallback_model,
                exc.status_code,
            )
            fallback_text, fallback_usage = self._fallback.complete(**kwargs)
            return fallback_text, fallback_usage, "fallback"
        if isinstance(text, str) and text.strip():
            return text, usage, "primary"
        logger.warning(
            "ACTOR_CARD_%s_FALLBACK primary_model=%s fallback_model=%s reason=empty_response",
            self._stage.upper(),
            self._primary_model,
            self._fallback_model,
        )
        fallback_text, fallback_usage = self._fallback.complete(**kwargs)
        return fallback_text, fallback_usage, "fallback"

    def complete_fallback(self, **kwargs):
        """Call the independent fallback directly for a coverage tiebreak."""
        text, usage = self._fallback.complete(**kwargs)
        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "ACTOR_CARD_%s_FALLBACK_EMPTY model=%s",
                self._stage.upper(),
                self._fallback_model,
            )
        return text, usage
