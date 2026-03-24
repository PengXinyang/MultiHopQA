from typing import Any, Dict, List


class RoleAwareLM:
    """
    角色路由语言模型：
    - 根据当前 role 将 query 转发到对应 LM 实例
    - 对外暴露与 AbstractLanguageModel 兼容的 query/get_response_texts 接口
    - 汇总各角色实例的 token/cost 统计
    """

    def __init__(self, role_to_lm: Dict[str, Any], default_role: str = "default") -> None:
        self.role_to_lm = role_to_lm
        self.default_role = default_role
        self.current_role = default_role
        self._last_role = default_role

    def set_role(self, role: str) -> None:
        if role and role in self.role_to_lm:
            self.current_role = role
        elif role == "critic_done" and "critic" in self.role_to_lm:
            self.current_role = "critic"
        else:
            self.current_role = self.default_role

    def _active_lm(self):
        role = self.current_role if self.current_role in self.role_to_lm else self.default_role
        self._last_role = role
        return self.role_to_lm[role]

    def query(self, query: str, num_responses: int = 1) -> Any:
        return self._active_lm().query(query, num_responses=num_responses)

    def get_response_texts(self, query_responses: Any) -> List[str]:
        role = self._last_role if self._last_role in self.role_to_lm else self.default_role
        return self.role_to_lm[role].get_response_texts(query_responses)

    @property
    def prompt_tokens(self) -> int:
        return int(sum(getattr(lm, "prompt_tokens", 0) for lm in self.role_to_lm.values()))

    @property
    def completion_tokens(self) -> int:
        return int(sum(getattr(lm, "completion_tokens", 0) for lm in self.role_to_lm.values()))

    @property
    def cost(self) -> float:
        return float(sum(getattr(lm, "cost", 0.0) for lm in self.role_to_lm.values()))
