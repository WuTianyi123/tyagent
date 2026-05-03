"""Tests for tyagent.model_metadata."""

from tyagent.model_metadata import (
    _strip_provider_prefix,
    get_model_context_length,
    FALLBACK_CONTEXT_LENGTH,
)


class TestStripProviderPrefix:
    def test_no_prefix(self):
        assert _strip_provider_prefix("deepseek-v4-pro") == "deepseek-v4-pro"

    def test_provider_prefix(self):
        assert _strip_provider_prefix("deepseek:deepseek-v4-pro") == "deepseek-v4-pro"

    def test_openrouter_prefix(self):
        assert _strip_provider_prefix("openrouter:anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"

    def test_ollama_tag_preserved(self):
        assert _strip_provider_prefix("qwen3.5:27b") == "qwen3.5:27b"

    def test_url_not_stripped(self):
        assert _strip_provider_prefix("https://api.example.com") == "https://api.example.com"


class TestGetModelContextLength:
    def test_explicit_override(self):
        """User-supplied context_length wins over table."""
        assert get_model_context_length("deepseek-v3", context_length=999_999) == 999_999

    def test_longest_substring_match(self):
        """deepseek-v4-pro matches before deepseek."""
        assert get_model_context_length("deepseek-v4-pro") == 1_000_000

    def test_family_catchall(self):
        """Unversioned model matches family key."""
        assert get_model_context_length("deepseek-some-future-model") == 128_000

    def test_case_insensitive(self):
        assert get_model_context_length("DEEPSEEK-V4-PRO") == 1_000_000

    def test_with_provider_prefix(self):
        """Provider prefix is stripped before lookup."""
        assert get_model_context_length("openrouter:deepseek-v4-pro") == 1_000_000

    def test_claude_hierarchy(self):
        """Specific claude models match before generic catchall."""
        assert get_model_context_length("claude-sonnet-4-6") == 1_000_000
        assert get_model_context_length("claude-sonnet-4") == 200_000
        assert get_model_context_length("claude-haiku-3") == 200_000  # "claude" catchall

    def test_gpt_hierarchy(self):
        assert get_model_context_length("gpt-5.5") == 1_050_000
        assert get_model_context_length("gpt-5.4-mini") == 400_000
        assert get_model_context_length("gpt-4o") == 128_000

    def test_unknown_model_fallback(self):
        assert get_model_context_length("non-existent-model-999") == FALLBACK_CONTEXT_LENGTH

    def test_deepseek_chat_aliases(self):
        """deepseek-chat and deepseek-reasoner now resolve to 1M (V4 family)."""
        assert get_model_context_length("deepseek-chat") == 1_000_000
        assert get_model_context_length("deepseek-reasoner") == 1_000_000

    def test_grok_4_1_fast(self):
        assert get_model_context_length("grok-4-1-fast-reasoning") == 2_000_000

    def test_cache_reuse(self):
        """Same model returns cached result."""
        r1 = get_model_context_length("claude-sonnet-4-6")
        r2 = get_model_context_length("claude-sonnet-4-6")
        assert r1 == r2 == 1_000_000
