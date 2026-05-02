"""
多组 Gemini API Key 并行与故障转移。

- 配置中同一逻辑模型可有多个带后缀的条目，如 gemini-2.5-flash-1 / -2 / -3。
- preferred_group_0based 指定优先组（0 -> …-1，1 -> …-2）；请求失败时按顺序尝试其余组。
- 仅用于原生 Gemini（非 -gcli）；与 factory.build_language_model 的 gemini_parallel_group_0based 配合使用。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

# 逻辑名（如 model_id / 角色配置里写的名字）-> config 里分组条目前缀
_DEFAULT_LOGICAL_TO_PREFIX: Dict[str, str] = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-flash-preview": "gemini-3-flash",
    "gemini-3-pro-preview": "gemini-3-pro",
    "gemini-3.1-pro-preview": "gemini-3-pro",
}


def _load_full_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {}
    return obj


def _aliases_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    block = cfg.get("gemini_native_parallel")
    if not isinstance(block, dict):
        return {}
    al = block.get("logical_model_aliases")
    if not isinstance(al, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in al.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _group_suffix_keys_for_prefix(cfg: Dict[str, Any], prefix: str) -> List[str]:
    """返回 config 中形如 ``{prefix}-{正整数}`` 的键，按数字升序。"""
    esc = re.escape(prefix)
    pat = re.compile(rf"^{esc}-(\d+)$")
    found: List[tuple[int, str]] = []
    for k in cfg.keys():
        if not isinstance(k, str):
            continue
        if k.endswith("-gcli") or "-gcli-" in k:
            continue
        m = pat.match(k)
        if m:
            found.append((int(m.group(1)), k))
    found.sort(key=lambda x: x[0])
    return [k for _, k in found]


def _all_gemini_group_suffix_keys(cfg: Dict[str, Any]) -> List[str]:
    """返回所有形如 ``gemini-...-{正整数}`` 的原生 Gemini 分组配置键。"""
    pat = re.compile(r"^gemini-.+-(\d+)$")
    found: List[tuple[str, int]] = []
    for k in cfg.keys():
        if not isinstance(k, str):
            continue
        if k.endswith("-gcli") or "-gcli-" in k:
            continue
        m = pat.match(k)
        if m:
            found.append((k, int(m.group(1))))
    found.sort(key=lambda x: (x[0].rsplit("-", 1)[0], x[1]))
    return [k for k, _ in found]


def resolve_gemini_config_prefix(logical_name: str, cfg: Dict[str, Any]) -> str:
    """
    将角色/工厂里写的 logical_name 解析为分组用的 config 键前缀（不含 -1/-2）。
    """
    name = (logical_name or "").strip()
    aliases = {**_DEFAULT_LOGICAL_TO_PREFIX, **_aliases_from_config(cfg)}
    if name in aliases:
        return aliases[name]

    m = re.match(r"^(.+)-(\d+)$", name)
    if m:
        base = m.group(1)
        if _group_suffix_keys_for_prefix(cfg, base):
            return base

    if _group_suffix_keys_for_prefix(cfg, name):
        return name

    return name


def ordered_config_keys_for_parallel(
    cfg: Dict[str, Any],
    logical_model_name: str,
    preferred_group_0based: int,
) -> List[str]:
    """
    返回按「优先组在前」排序的一组 config 键；每组对应一个 api_key。
    """
    prefix = resolve_gemini_config_prefix(logical_model_name, cfg)
    keys = _group_suffix_keys_for_prefix(cfg, prefix)
    if not keys:
        return [logical_model_name]

    n = len(keys)
    p = int(preferred_group_0based) % n
    order = [(p + i) % n for i in range(n)]
    return [keys[i] for i in order]


def parallel_gemini_groups_enabled(config_path: str) -> bool:
    """是否在并行进程池中为子进程分配不同的 Gemini key 组（可由 config 关闭）。"""
    cfg = _load_full_config(config_path)
    block = cfg.get("gemini_native_parallel")
    if not isinstance(block, dict):
        return True
    return bool(block.get("assign_key_group_by_process", True))


def gemini_parallel_groups_configured(config_path: str) -> bool:
    """config 中是否实际存在原生 Gemini 分组键。"""
    cfg = _load_full_config(config_path)
    return bool(_all_gemini_group_suffix_keys(cfg))


def detect_gemini_parallel_num_groups(config_path: str) -> int:
    """根据 config 中 gemini-2.5-flash-* 等条目推断组数；无法推断时返回 1。"""
    cfg = _load_full_config(config_path)
    block = cfg.get("gemini_native_parallel")
    if isinstance(block, dict) and block.get("num_groups") is not None:
        try:
            ng = int(block["num_groups"])
            return max(1, ng)
        except (TypeError, ValueError):
            pass
    keys = _group_suffix_keys_for_prefix(cfg, "gemini-2.5-flash")
    return max(1, len(keys))


class GeminiGroupedFailover:
    """
    同一逻辑模型多组 API key：优先使用 preferred 组，失败则依次尝试其它组。
    """

    def __init__(
        self,
        config_path: str,
        logical_model_name: str,
        preferred_group_0based: int,
        cache: bool = False,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path
        self.logical_model_name = logical_model_name.strip()
        self.model_name = self.logical_model_name
        self.cache = bool(cache)
        if self.cache:
            self.response_cache: Dict[str, Any] = {}

        full = _load_full_config(config_path)
        ordered_keys = ordered_config_keys_for_parallel(
            full, self.logical_model_name, preferred_group_0based
        )
        self._ordered_keys: List[str] = list(ordered_keys)
        self._backends: List[Any] = []
        from .gemini import Gemini

        for k in self._ordered_keys:
            self._backends.append(
                Gemini(
                    config_path,
                    model_name=k,
                    cache=cache,
                    ignore_env_api_key=True,
                )
            )
        self._last_backend: Optional[Gemini] = None

    def set_role(self, role: str) -> None:
        for b in self._backends:
            if hasattr(b, "set_role"):
                try:
                    b.set_role(role)
                except Exception:
                    continue

    @property
    def prompt_tokens(self) -> int:
        return int(sum(getattr(b, "prompt_tokens", 0) for b in self._backends))

    @property
    def completion_tokens(self) -> int:
        return int(sum(getattr(b, "completion_tokens", 0) for b in self._backends))

    @property
    def cost(self) -> float:
        return float(sum(getattr(b, "cost", 0.0) for b in self._backends))

    def query(self, query: str, num_responses: int = 1) -> Union[List[Any], Any]:
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        last_exc: Optional[BaseException] = None
        for b in self._backends:
            try:
                res = b.query(query, num_responses=num_responses)
                texts = b.get_response_texts(res)
                if not isinstance(texts, list) or not any(
                    isinstance(t, str) and t.strip() for t in texts
                ):
                    raise RuntimeError("empty_response_texts")
                self._last_backend = b
                if self.cache:
                    self.response_cache[query] = res
                return res
            except Exception as e:
                last_exc = e
                self.logger.warning(
                    "Gemini 组切换: config_key=%s 失败 (%s)，尝试下一组 key",
                    getattr(b, "model_name", "?"),
                    e,
                )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("GeminiGroupedFailover: 无可用后端")

    def get_response_texts(self, query_response: Union[List[Any], Any]) -> List[str]:
        b = self._last_backend
        if b is not None:
            return b.get_response_texts(query_response)
        if self._backends:
            return self._backends[0].get_response_texts(query_response)
        return []
