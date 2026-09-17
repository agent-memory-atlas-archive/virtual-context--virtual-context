

from virtual_context.core.community.actor_card_curation import ActorCardCurationService
from virtual_context.providers.anthropic import AnthropicProvider
from virtual_context.providers.generic_openai import GenericOpenAIProvider
from virtual_context.types import VirtualContextConfig


def test_anthropic_provider_base_url_and_thinking_switch():
    p = AnthropicProvider(api_key="k", model="k3", base_url="https://api.kimi.com/coding/", disable_thinking=True)
    assert p._get_url() == "https://api.kimi.com/coding/v1/messages"
    payload = p._build_payload("sys", "user", 64)
    assert payload["thinking"] == {"type": "disabled"} and payload["model"] == "k3"
    default = AnthropicProvider(api_key="k")
    assert default._get_url().startswith("https://api.anthropic.com") and "thinking" not in default._build_payload("s", "u", 8)


class _Compactor:
    def __init__(self, llm):
        self.llm = llm


def _service(config, llm):
    return ActorCardCurationService(
        config=config, compactor=_Compactor(llm), prompt_turns=lambda *a, **k: [],
        curation_provider=lambda: None, provider_for_model=lambda m: None,
    )


def test_named_provider_routes_prefixed_model_to_declared_gateway(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "kimi-secret")
    cfg = VirtualContextConfig()
    cfg.providers = {"kimi": {"type": "anthropic", "base_url": "https://api.kimi.com/coding",
                              "api_key_env": "KIMI_API_KEY", "disable_thinking": True}}
    base = GenericOpenAIProvider(base_url="https://openrouter.ai/api/v1", model="google/gemini-2.5-flash-lite", api_key="or-key")
    svc = _service(cfg, base)
    p = svc.provider_for_model("kimi/k3")
    assert isinstance(p, AnthropicProvider) and p.model == "k3" and p.api_key == "kimi-secret"
    assert p._get_url() == "https://api.kimi.com/coding/v1/messages" and p.disable_thinking is True
    assert p.temperature == 0.0


def test_unprefixed_or_unknown_prefix_keeps_the_summarizer_gateway():
    cfg = VirtualContextConfig()
    cfg.providers = {"kimi": {"type": "anthropic", "api_key": "x"}}
    base = GenericOpenAIProvider(base_url="https://openrouter.ai/api/v1", model="google/gemini-2.5-flash-lite", api_key="or-key")
    svc = _service(cfg, base)
    p = svc.provider_for_model("qwen/qwen3-235b-a22b-2507")
    assert isinstance(p, GenericOpenAIProvider) and p.model == "qwen/qwen3-235b-a22b-2507" and p.base_url == base.base_url
