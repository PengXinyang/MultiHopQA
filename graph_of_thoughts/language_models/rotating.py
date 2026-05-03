"""
Rotating / failover language model wrappers.

- Retry each underlying model N times on exception.
- If still failing, switch to next model and continue.

This wrapper is intentionally lightweight: it delegates token/cost accounting
to the underlying models and exposes aggregate properties.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .abstract_language_model import AbstractLanguageModel
from .factory import build_language_model


@dataclass(frozen=True)
class RotatingModelConfig:
    model_names: List[str]
    retries_per_model: int = 3
    cache: bool = True
    gemini_parallel_group_0based: Optional[int] = None


class RotatingLanguageModel:
    """
    A drop-in wrapper (duck-typed) for AbstractLanguageModel usage:
    - query()
    - get_response_texts()
    - optional set_role()
    - prompt_tokens/completion_tokens/cost aggregated
    """

    def __init__(self, config_path: str, config: RotatingModelConfig) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        self.model_names: List[str] = [m.strip() for m in (config.model_names or []) if str(m).strip()]
        if not self.model_names:
            raise ValueError("RotatingLanguageModel requires a non-empty model_names list")

        self.retries_per_model = int(config.retries_per_model or 3)
        self.cache = bool(config.cache)
        self._gemini_parallel_group_0based: Optional[int] = config.gemini_parallel_group_0based

        self._models: List[Any] = []
        kept_names: List[str] = []
        for name in self.model_names:
            try:
                self._models.append(
                    build_language_model(
                        config_path,
                        model_name=name,
                        cache=self.cache,
                        gemini_parallel_group_0based=self._gemini_parallel_group_0based,
                    )
                )
                kept_names.append(name)
            except Exception as e:
                # If a model cannot be constructed (missing key/env/dependency), skip it.
                self.logger.warning("Skip model init (model=%s): %s", name, repr(e))

        self.model_names = kept_names
        if not self._models:
            raise ValueError("RotatingLanguageModel could not initialize any underlying model")

        self._active_idx: int = 0
        self._last_success_idx: Optional[int] = None

    def set_role(self, role: str) -> None:
        # Forward role to underlying models (if they support it).
        for m in self._models:
            if hasattr(m, "set_role"):
                try:
                    m.set_role(role)
                except Exception:
                    # Role forwarding should never break the call path.
                    continue

    def _active_model(self):
        return self._models[self._active_idx]

    def _advance(self) -> None:
        self._active_idx = (self._active_idx + 1) % len(self._models)

    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        Try current model up to retries_per_model times.
        On repeated failure, rotate to next model.
        """
        last_exc: Optional[BaseException] = None
        attempted_models = 0
        while attempted_models < len(self._models):
            m = self._active_model()
            model_name = getattr(m, "model_name", None) or self.model_names[self._active_idx]
            for attempt in range(1, self.retries_per_model + 1):
                try:
                    res = m.query(query, num_responses=num_responses)
                    # Treat "empty response / no extractable text" as a failure too.
                    try:
                        texts = m.get_response_texts(res)
                    except Exception as e:
                        raise RuntimeError(f"response_parse_failed: {e!r}") from e
                    if not isinstance(texts, list) or not any(isinstance(t, str) and t.strip() for t in texts):
                        raise RuntimeError("empty_response_texts")

                    self._last_success_idx = self._active_idx
                    return res
                except Exception as e:
                    last_exc = e
                    self.logger.warning(
                        "Model query failed (model=%s attempt=%d/%d): %s",
                        model_name,
                        attempt,
                        self.retries_per_model,
                        repr(e),
                    )
            # switch model after exhausting retries
            self._advance()
            attempted_models += 1

        # All models failed
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("All models failed, but no exception was captured")

    def get_response_texts(self, query_responses: Any) -> List[str]:
        # Use the most recent successful model if known; otherwise fall back to current.
        idx = self._last_success_idx if self._last_success_idx is not None else self._active_idx
        return self._models[idx].get_response_texts(query_responses)

    @property
    def prompt_tokens(self) -> int:
        return int(sum(getattr(m, "prompt_tokens", 0) for m in self._models))

    @property
    def completion_tokens(self) -> int:
        return int(sum(getattr(m, "completion_tokens", 0) for m in self._models))

    @property
    def cost(self) -> float:
        return float(sum(getattr(m, "cost", 0.0) for m in self._models))


def _load_model_keys_from_config(config_path: str) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return []
    return [str(k) for k in obj.keys()]


class LightweightModelGroup(RotatingLanguageModel):
    """
    Lightweight model pool (fixed list).

    Note: model names here refer to keys in `graph_of_thoughts/language_models/config.json`,
    not provider-native ids.
    """

    LITE_MODELS: List[str] = [
        #"chatgpt",  # gpt-3.5-turbo
        "deepseek-v4-flash",
        "gemini-2.5-flash-gcli",
        "gemini-2.5-flash-1",
        # "gemini-2.5-flash-2",
        # "gemini-3-flash-1",
        # "gemini-3-flash-2",
    ]

    #: 并行且按进程分配 Gemini key 组时：每角色内为逻辑名，组内先 2.5-flash 再 3-flash，组间由 GeminiGroupedFailover 切换
    LITE_MODELS_PARALLEL: List[str] = [
        "gemini-2.5-flash",
        "gemini-3-flash",
    ]

    def __init__(
        self,
        config_path: str,
        cache: bool = True,
        retries_per_model: int = 3,
        gemini_parallel_group_0based: Optional[int] = None,
    ) -> None:
        names = (
            list(self.LITE_MODELS_PARALLEL)
            if gemini_parallel_group_0based is not None
            else list(self.LITE_MODELS)
        )
        super().__init__(
            config_path=config_path,
            config=RotatingModelConfig(
                model_names=names,
                retries_per_model=retries_per_model,
                cache=cache,
                gemini_parallel_group_0based=gemini_parallel_group_0based,
            ),
        )


class HeavyModelGroup(RotatingLanguageModel):
    """
    Heavy/complex model pool.

    By default, this uses all models from config.json excluding the lightweight pool.
    You can also pass an explicit model list.
    """

    HEAVY_MODELS_PARALLEL: List[str] = [
        "gemini-2.5-pro",
        "gemini-3-pro",
    ]

    def __init__(
        self,
        config_path: str,
        cache: bool = True,
        retries_per_model: int = 3,
        model_names: Optional[Sequence[str]] = None,
        gemini_parallel_group_0based: Optional[int] = None,
    ) -> None:
        if model_names is None:
            if gemini_parallel_group_0based is not None:
                model_names = list(self.HEAVY_MODELS_PARALLEL)
            else:
                model_names = [
                    #"chatgpt4",
                    "deepseek-v4-pro",
                    "gemini-2.5-pro-gcli",
                    #"gemini-3-pro-gcli",
                    "gemini-2.5-pro-1",
                    # "gemini-2.5-pro-2",
                    # "gemini-3-pro-1",
                    # "gemini-3-pro-2",
                ]
        super().__init__(
            config_path=config_path,
            config=RotatingModelConfig(
                model_names=list(model_names),
                retries_per_model=retries_per_model,
                cache=cache,
                gemini_parallel_group_0based=gemini_parallel_group_0based,
            ),
        )

