"""LLM client factory — any OpenAI-compatible provider (NVIDIA / Groq / Gemini / Ollama)."""

from langchain_openai import ChatOpenAI

from .config import (
    CHAT_PROVIDER_ORDER,
    CONFIG,
    PROVIDERS,
    ProviderSpec,
    get_api_key,
)


def pick_chat_provider(preferred: str | None = None) -> ProviderSpec:
    if preferred and preferred in PROVIDERS:
        p = PROVIDERS[preferred]
        if p.chat_models:
            if not p.api_key_env or get_api_key(p):
                return p
            raise RuntimeError(f"Provider '{preferred}' selected but {p.api_key_env} is not set.")
    for key in CHAT_PROVIDER_ORDER:
        p = PROVIDERS[key]
        if p.chat_models and (not p.api_key_env or get_api_key(p)):
            return p
    raise RuntimeError(
        "No chat provider configured. Set GOOGLE_API_KEY, GROQ_API_KEY or "
        "NVIDIA_API_KEY in backend/.env — or start Ollama locally."
    )


def _build_chat_llm(p: ProviderSpec, model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=get_api_key(p) or "not-needed",
        base_url=p.base_url,
        temperature=temperature,
        max_tokens=CONFIG.max_tokens,
        # Fail fast so provider failover happens quickly (e.g. NVIDIA 504s)
        timeout=CONFIG.llm_timeout_seconds,
        max_retries=CONFIG.llm_max_retries,
    )


class FallbackChatLLM:
    """Chat wrapper with automatic provider failover.

    Tries providers in preference order (Google -> NVIDIA -> Ollama).
    If a call fails (quota exhausted, 429, 403, network...), the next
    provider is used. The last successful provider is remembered and
    tried first on subsequent calls."""

    def __init__(self, temperature: float,
                 preferred_provider: str | None = None,
                 preferred_model: str | None = None) -> None:
        self.temperature = temperature
        self.preferred_provider = preferred_provider
        self.preferred_model = preferred_model
        self._last_working: tuple[str, str] | None = None

    def _candidates(self) -> list[tuple[ProviderSpec, str]]:
        out: list[tuple[ProviderSpec, str]] = []
        # 1) explicit user choice first (dropdown selection)
        if self.preferred_provider and self.preferred_provider in PROVIDERS:
            p = PROVIDERS[self.preferred_provider]
            if p.chat_models:
                model = self.preferred_model if self.preferred_model in p.chat_models \
                    else p.chat_models[0]
                out.append((p, model))
        # 2) last working combo (sticky)
        if self._last_working:
            key, model = self._last_working
            p = PROVIDERS.get(key)
            if p and (p, model) not in out:
                out.insert(1 if self.preferred_provider else 0, (p, model))
        # 3) configured order
        for key in CHAT_PROVIDER_ORDER:
            p = PROVIDERS[key]
            if p.chat_models and (not p.api_key_env or get_api_key(p)):
                for m in p.chat_models:
                    if all(pk.key != key or mm != m for pk, mm in out):
                        out.append((p, m))
                        break
        return out

    def invoke(self, messages):
        errors: list[str] = []
        for p, model in self._candidates():
            try:
                llm = _build_chat_llm(p, model, self.temperature)
                result = llm.invoke(messages)
                if self._last_working != (p.key, model):
                    print(f"[llm] Using {p.label} ({model})")
                self._last_working = (p.key, model)
                return result
            except Exception as exc:                     # noqa: BLE001 - try next provider
                errors.append(f"{p.key}/{model}: {str(exc)[:150]}")
        raise RuntimeError(
            "All chat providers failed -> " + " | ".join(errors))

    @staticmethod
    def _chunk_text(chunk) -> str:
        content = getattr(chunk, "content", chunk)
        if isinstance(content, str):
            return content
        if isinstance(content, list):                    # some providers send parts
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                           for part in content)
        return str(content or "")

    def stream(self, messages):
        """Yield answer chunks token-by-token with the same failover as invoke().
        Falls to the next provider only if NO token arrived yet from this one."""
        errors: list[str] = []
        for p, model in self._candidates():
            try:
                llm = _build_chat_llm(p, model, self.temperature)
                iterator = llm.stream(messages)
                first = next(iterator)                   # may raise -> next provider
            except StopIteration:
                return
            except Exception as exc:                     # noqa: BLE001
                errors.append(f"{p.key}/{model}: {str(exc)[:150]}")
                continue
            if self._last_working != (p.key, model):
                print(f"[llm] Using {p.label} ({model}) [stream]")
            self._last_working = (p.key, model)
            yield self._chunk_text(first)
            try:
                for chunk in iterator:
                    yield self._chunk_text(chunk)
            except Exception:                            # mid-stream drop: stop gracefully
                return
            return
        raise RuntimeError(
            "All chat providers failed -> " + " | ".join(errors))


def get_llm(temperature: float | None = None,
            provider: str | None = None,
            model: str | None = None) -> FallbackChatLLM:
    """Return a chat LLM with automatic Google -> NVIDIA failover."""
    temperature = CONFIG.answer_temperature if temperature is None else temperature
    return FallbackChatLLM(temperature, provider, model)
