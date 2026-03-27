"""
操作模块：定义 Graph of Thoughts 中的各种操作类型。

包含的操作：
- Generate: 生成新思维
- Score: 对思维评分
- KeepBestN: 保留得分最高的 N 个思维
- Aggregate: 聚合多个思维
- ValidateAndImprove: 验证并改进思维
- Improve: 改进思维
- KeepValid: 保留有效思维
- GroundTruth: 与标准答案比较
- Selector: 选择特定思维
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import List, Iterator, Dict, Callable, Union
from abc import ABC, abstractmethod
import itertools

from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser


class OperationType(Enum):
    """
    操作类型枚举，用作操作的唯一标识符。
    """
    score: int = 0  # 评分
    validate_and_improve: int = 1  # 验证并改进
    generate: int = 2  # 生成
    improve: int = 3  # 改进
    aggregate: int = 4  # 聚合
    keep_best_n: int = 5  # 保留最佳 N 个
    keep_valid: int = 6  # 保留有效
    ground_truth_evaluator: int = 7  # 标准答案评估
    selector: int = 8  # 选择器
    critic_verify_and_backtrack: int = 9  # 带有回溯机制的评估器
    advance_subquestion: int = 10  # multiAgentGoT: 推进到下一跳子问题

class BacktrackSignal(Exception):
    """用于触发全局回溯的异常信号"""
    def __init__(self, target_operation: Operation, reason: str = ""):
        self.target_operation = target_operation
        self.reason = reason


class Operation(ABC):
    """
    操作的抽象基类，定义所有操作的公共接口。
    """

    _ids: Iterator[int] = itertools.count(0)
    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        初始化一个新的 Operation 实例，分配唯一 ID，前驱和后继列表为空。
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []  # 前驱操作列表
        self.successors: List[Operation] = []  # 后继操作列表
        self.executed: bool = False  # 是否已执行

    def can_be_executed(self) -> bool:
        """
        检查操作是否可以执行（所有前驱操作都已执行）。

        :return: 如果所有前驱已执行返回 True，否则返回 False
        :rtype: bool
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        """
        获取所有前驱操作的思维列表。

        :return: 所有前驱操作的思维汇总列表
        :rtype: List[Thought]
        """
        previous_thoughts: List[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]
        return previous_thoughts

    def add_predecessor(self, operation: Operation) -> None:
        """
        添加前驱操作，并更新双向关系。

        :param operation: 要设为前驱的操作
        :type operation: Operation
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: Operation) -> None:
        """
        添加后继操作，并更新双向关系。

        :param operation: 要设为后继的操作
        :type operation: Operation
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行操作，确保所有前驱已执行。

        :param lm: 语言模型
        :type lm: AbstractLanguageModel
        :param prompter: 提示词生成器
        :type prompter: Prompter
        :param parser: 响应解析器
        :type parser: Parser
        :param kwargs: 额外参数
        :raises AssertionError: 如果前驱未全部执行
        """
        assert self.can_be_executed(), "前驱操作尚未全部执行"
        self.logger.info("正在执行操作 %d，类型 %s", self.id, self.operation_type)
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug("操作 %d 执行完成", self.id)
        self.executed = True

    @abstractmethod
    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        操作的实际执行逻辑（抽象方法，由子类实现）。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        """
        获取操作关联的思维列表（抽象方法，由子类实现）。

        :return: 思维列表
        :rtype: List[Thought]
        """
        pass


class Score(Operation):
    """
    评分操作：对思维进行评分。
    
    可以使用 LLM 评分或自定义评分函数。
    支持单独评分或组合评分模式。
    """

    operation_type: OperationType = OperationType.score

    def __init__(
            self,
            num_samples: int = 1,
            combined_scoring: bool = False,
            scoring_function: Callable[
                [Union[List[Dict], Dict]], Union[List[float], float]
            ] = None,
    ) -> None:
        """
        初始化 Score 操作。

        :param num_samples: LLM 评分的采样次数，默认为 1
        :type num_samples: int
        :param combined_scoring: 是否对所有思维一起评分，默认为 False（单独评分）
        :type combined_scoring: bool
        :param scoring_function: 自定义评分函数（不使用 LLM），默认为 None
        :type scoring_function: Callable，接收思维状态（列表或单个），返回评分（列表或单个）
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.thoughts: List[Thought] = []
        self.scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = scoring_function

    def get_thoughts(self) -> List[Thought]:
        """
        返回评分后的思维列表。

        :return: 已评分的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行评分操作。
        
        如果使用组合评分，所有思维一起评分；否则单独评分。
        如果提供了评分函数则使用它，否则调用 LLM。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
                len(self.predecessors) > 0
        ), "Score 操作需要至少一个前驱操作"

        if self.combined_scoring:
            # 组合评分：所有思维一起评分
            previous_thoughts_states = [thought.state for thought in previous_thoughts]
            if self.scoring_function is not None:
                self.logger.debug("使用评分函数 %s 评分", self.scoring_function)
                scores = self.scoring_function(previous_thoughts_states)
            else:
                if hasattr(lm, "set_role"):
                    role = "default"
                    if previous_thoughts_states and isinstance(previous_thoughts_states[0], dict):
                        role = previous_thoughts_states[0].get("agent_role", "default")
                        if previous_thoughts_states[0].get("method", "").startswith("multiAgentGoT"):
                            role = "critic"
                    lm.set_role(role)
                prompt = prompter.score_prompt(previous_thoughts_states)
                self.logger.debug("LLM 提示词: %s", prompt)
                responses = lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_samples)
                )
                self.logger.debug("LLM 响应: %s", responses)
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
            for thought, score in zip(previous_thoughts, scores):
                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                self.thoughts.append(new_thought)
        else:
            # 单独评分：逐个思维评分
            for thought in previous_thoughts:
                new_thought = Thought.from_thought(thought)
                if self.scoring_function is not None:
                    self.logger.debug("使用评分函数 %s 评分", self.scoring_function)
                    score = self.scoring_function(thought.state)
                else:
                    if hasattr(lm, "set_role"):
                        role = thought.state.get("agent_role", "default")
                        if thought.state.get("method", "").startswith("multiAgentGoT"):
                            role = "critic"
                        lm.set_role(role)
                    prompt = prompter.score_prompt([thought.state])
                    self.logger.debug("LLM 提示词: %s", prompt)
                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug("LLM 响应: %s", responses)
                    score = parser.parse_score_answer([thought.state], responses)[0]
                new_thought.score = score
                # multiAgentGoT 在最终答案评分阶段记录全局评价
                try:
                    if (
                        thought.state.get("method", "").startswith("multiAgentGoT")
                        and (thought.state.get("answer") or "").strip()
                        and hasattr(parser, "parse_score_critique")
                    ):
                        critique = parser.parse_score_critique(thought.state, responses)
                        if critique:
                            new_thought.state = dict(new_thought.state or {})
                            new_thought.state["global_critique"] = critique
                except Exception:
                    pass
                self.thoughts.append(new_thought)

        self.logger.info("Score 操作 %d 对 %d 个思维评分完成", self.id, len(self.thoughts))


class ValidateAndImprove(Operation):
    """
    验证并改进操作：验证思维的有效性，如果无效则尝试改进。
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(
            self,
            num_samples: int = 1,
            improve: bool = True,
            num_tries: int = 3,
            validate_function: Callable[[Dict], bool] = None,
    ) -> None:
        """
        初始化 ValidateAndImprove 操作。

        :param num_samples: 验证的采样次数，默认为 1
        :type num_samples: int
        :param improve: 如果无效是否尝试改进，默认为 True
        :type improve: bool
        :param num_tries: 改进的最大尝试次数，默认为 3
        :type num_tries: int
        :param validate_function: 自定义验证函数，默认为 None（使用 LLM）
        :type validate_function: Callable，接收思维状态，返回布尔值
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.improve: bool = improve
        self.num_tries: int = num_tries
        self.validate_function: Callable[[Dict], bool] = validate_function
        self.thoughts: List[List[Thought]] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回验证和改进后的最终思维列表。

        :return: 每个思维链的最后一个思维
        :rtype: List[Thought]
        """
        return [thought_list[-1] for thought_list in self.thoughts]

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行验证并改进操作。
        
        对每个前驱思维：验证 -> 如果无效且启用改进 -> 改进 -> 重复直到有效或达到最大尝试次数。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
                len(self.predecessors) > 0
        ), "ValidateAndImprove 操作需要至少一个前驱操作"

        for thought in previous_thoughts:
            thought_list = []
            current_thought = Thought.from_thought(thought)
            current_try = 0
            while True:
                # 验证
                if self.validate_function is not None:
                    self.logger.debug("使用验证函数 %s", self.validate_function)
                    valid = self.validate_function(current_thought.state)
                else:
                    prompt = prompter.validation_prompt(**current_thought.state)
                    self.logger.debug("LLM 提示词: %s", prompt)
                    responses = lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_samples)
                    )
                    self.logger.debug("LLM 响应: %s", responses)
                    valid = parser.parse_validation_answer(current_thought.state, responses)

                current_thought.valid = valid
                thought_list.append(current_thought)

                # 检查是否需要继续改进
                if not self.improve or current_thought.valid or current_try >= self.num_tries:
                    break

                # 改进
                improve_prompt = prompter.improve_prompt(**current_thought.state)
                self.logger.debug("LLM 提示词: %s", improve_prompt)
                responses = lm.get_response_texts(lm.query(improve_prompt, num_responses=1))
                self.logger.debug("LLM 响应: %s", responses)
                state_update = parser.parse_improve_answer(current_thought.state, responses)
                current_thought = Thought({**current_thought.state, **state_update})
                current_try += 1

            self.thoughts.append(thought_list)

        self.logger.info(
            "ValidateAndImprove 操作 %d 从 %d 个思维中创建了 %d 个有效思维",
            self.id,
            len(previous_thoughts),
            len([tl[-1] for tl in self.thoughts if tl[-1].valid]),
        )


class Generate(Operation):
    """
    生成操作：基于前驱思维生成新的思维。
    
    这是最常用的操作之一，用于调用 LLM 生成新的推理步骤。
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
            self, num_branches_prompt: int = 1, num_branches_response: int = 1
    ) -> None:
        """
        初始化 Generate 操作。

        :param num_branches_prompt: 提示词要求生成的响应数量（传给 prompter），默认为 1
        :type num_branches_prompt: int
        :param num_branches_response: LLM 实际生成的响应数量，默认为 1
        :type num_branches_response: int
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回生成的思维列表。

        :return: 生成的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行生成操作。
        
        使用前驱思维的状态构造提示词，调用 LLM 生成响应，解析为新思维。
        如果没有前驱操作，使用 kwargs 作为基础状态。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数（作为初始状态）
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        # 如果前驱存在但没有思维，直接返回
        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        # 如果没有前驱，使用 kwargs 作为基础状态
        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(state=kwargs)]

        pp_ref = kwargs.get("__pp_ref")
        retry_by_hop = {}
        if isinstance(pp_ref, dict):
            retry_by_hop = pp_ref.get("retry_feedback_by_hop") or {}
            if not isinstance(retry_by_hop, dict):
                retry_by_hop = {}

        for thought in previous_thoughts:
            # base_state = dict(thought.state or {})
            # # 如果当前 Generate 节点带有 node_instruction（来自 AI GoO），
            # # 将其注入到状态中，供 Prompter 动态构造提示词使用。
            # node_instruction = getattr(self, "node_instruction", None)
            # if node_instruction:
            #     base_state["node_instruction"] = node_instruction
            base_state = thought.state
            # Inject persisted retry feedback into multiAgentGoT prompts (kept across backtracking).
            try:
                if (
                    isinstance(base_state, dict)
                    and str(base_state.get("method", "")).startswith("multiAgentGoT")
                    and base_state.get("agent_role") in ("retriever", "reasoner")
                ):
                    sid = int(base_state.get("sub_id", -1) or -1)
                    fb = retry_by_hop.get(str(sid)) or retry_by_hop.get(sid)
                    if isinstance(fb, dict) and fb:
                        # Do not overwrite any newer in-state feedback.
                        base_state.setdefault("retry_feedback", fb)
            except Exception:
                pass
            if hasattr(lm, "set_role"):
                lm.set_role(base_state.get("agent_role", "default"))
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug("LLM 提示词: %s", prompt)
            responses = lm.get_response_texts(
                lm.query(prompt, num_responses=self.num_branches_response)
            )
            self.logger.debug("LLM 响应: %s", responses)

            for new_state in parser.parse_generate_answer(base_state, responses):
                new_state = {**base_state, **new_state}
                self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    "创建新思维 %d，状态: %s", self.thoughts[-1].id, self.thoughts[-1].state
                )

        # 警告：生成的思维数量超出预期
        if (
                len(self.thoughts)
                > self.num_branches_prompt * self.num_branches_response * len(previous_thoughts)
                and self.num_branches_prompt > 0
        ):
            self.logger.warning("Generate 操作 %d 创建的思维数量超出预期", self.id)

        self.logger.info("Generate 操作 %d 创建了 %d 个新思维", self.id, len(self.thoughts))


class Improve(Operation):
    """
    改进操作：对思维进行改进。
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        """
        初始化 Improve 操作。
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回改进后的思维列表。

        :return: 改进后的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行改进操作。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert len(self.predecessors) > 0, "Improve 操作需要至少一个前驱操作"

        for thought in previous_thoughts:
            improve_prompt = prompter.improve_prompt(**thought.state)
            self.logger.debug("LLM 提示词: %s", improve_prompt)
            responses = lm.get_response_texts(lm.query(improve_prompt, num_responses=1))
            self.logger.debug("LLM 响应: %s", responses)
            state_update = parser.parse_improve_answer(thought.state, responses)
            self.thoughts.append(Thought({**thought.state, **state_update}))

        self.logger.info("Improve 操作 %d 改进了 %d 个思维", self.id, len(self.thoughts))


class Aggregate(Operation):
    """
    聚合操作：将多个思维聚合为一个或多个新思维。
    
    常用于合并多个分支的推理结果。
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, num_responses: int = 1) -> None:
        """
        初始化 Aggregate 操作。

        :param num_responses: 聚合生成的响应数量，默认为 1
        :type num_responses: int
        """
        super().__init__()
        self.thoughts: List[Thought] = []
        self.num_responses: int = num_responses

    def get_thoughts(self) -> List[Thought]:
        """
        返回聚合后的思维列表。

        :return: 聚合后的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行聚合操作。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        assert (
                len(self.predecessors) >= 1
        ), "Aggregate 操作需要至少一个前驱操作"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            return

        # 按评分排序合并状态（评分高的覆盖评分低的）
        base_state: Dict = {}
        for thought in sorted(previous_thoughts, key=lambda t: t.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]
        prompt = prompter.aggregation_prompt(previous_thought_states)

        self.logger.debug("LLM 提示词: %s", prompt)

        responses = lm.get_response_texts(
            lm.query(prompt, num_responses=self.num_responses)
        )

        self.logger.debug("LLM 响应: %s", responses)

        parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

        if isinstance(parsed, dict):
            parsed = [parsed]
        for new_state in parsed:
            self.thoughts.append(Thought({**base_state, **new_state}))


class KeepBestN(Operation):
    """
    保留最佳 N 个操作：根据评分保留得分最高（或最低）的 N 个思维。
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        """
        初始化 KeepBestN 操作。

        :param n: 要保留的思维数量
        :type n: int
        :param higher_is_better: True 表示分数越高越好，False 表示分数越低越好
        :type higher_is_better: bool
        :raises AssertionError: 如果 n <= 0
        """
        super().__init__()
        self.n: int = n
        assert self.n > 0, "KeepBestN 操作必须保留至少一个思维"
        self.higher_is_better: bool = higher_is_better
        self.thoughts: List[Thought] = []

    def get_best_n(self) -> List[Thought]:
        """
        获取评分最好的 N 个思维。

        :return: 最佳 N 个思维列表
        :rtype: List[Thought]
        :raises AssertionError: 如果思维未全部评分
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()
        assert all(
            t.scored for t in previous_thoughts
        ), "并非所有思维都已评分"

        try:
            return sorted(
                previous_thoughts,
                key=lambda t: t.score,
                reverse=self.higher_is_better,
            )[: self.n]
        except:
            self.logger.error("KeepBestN 操作出错")
            self.logger.error("前驱操作: %s", [op.id for op in self.predecessors])
            self.logger.error("前驱思维: %s", previous_thoughts)
            self.logger.error("评分: %s", [t.score for t in previous_thoughts])
            return sorted(
                [t for t in previous_thoughts if isinstance(t.score, float)],
                key=lambda t: t.score,
                reverse=self.higher_is_better,
            )[: self.n]

    def get_thoughts(self) -> List[Thought]:
        """
        返回保留的思维列表。

        :return: 保留的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行 KeepBestN 操作。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        assert (
                len(self.predecessors) >= 1
        ), "KeepBestN 操作需要至少一个前驱操作"

        self.thoughts = [Thought.from_thought(t) for t in self.get_best_n()]

        for thought in self.thoughts:
            self.logger.debug("保留思维 %d，状态: %s", thought.id, thought.state)

        self.logger.info("KeepBestN 操作 %d 保留了 %d 个思维", self.id, len(self.thoughts))


class KeepValid(Operation):
    """
    保留有效思维操作：仅保留验证通过的思维（未验证的也保留）。
    """

    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        """
        初始化 KeepValid 操作。
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回保留的思维列表。

        :return: 保留的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行 KeepValid 操作。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        assert (
                len(self.predecessors) >= 1
        ), "KeepValid 操作需要至少一个前驱操作"

        self.thoughts: List[Thought] = [
            Thought.from_thought(thought)
            for thought in self.get_previous_thoughts()
            if not thought.validated or thought.valid
        ]

        if any(not thought.validated for thought in self.thoughts):
            self.logger.warning("KeepValid 操作 %d 包含未验证的思维", self.id)

        for thought in self.thoughts:
            self.logger.debug("保留思维 %d，状态: %s", thought.id, thought.state)

        self.logger.info("KeepValid 操作 %d 保留了 %d 个思维", self.id, len(self.thoughts))


class GroundTruth(Operation):
    """
    标准答案评估操作：使用标准答案评估函数检查思维是否正确解决了问题。
    """

    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(self, ground_truth_evaluator: Callable[[Dict], bool]) -> None:
        """
        初始化 GroundTruth 操作。

        :param ground_truth_evaluator: 评估函数，接收思维状态，返回是否解决问题
        :type ground_truth_evaluator: Callable[[Dict], bool]
        """
        super().__init__()
        self.ground_truth_evaluator: Callable[[Dict], bool] = ground_truth_evaluator
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回评估后的思维列表。

        :return: 评估后的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行 GroundTruth 评估操作。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        :raises AssertionError: 如果没有前驱操作
        """
        assert (
                len(self.predecessors) >= 1
        ), "GroundTruth 操作需要至少一个前驱操作"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            try:
                eval_state = {**(new_thought.state or {}), "_thought_score": new_thought.score}
                new_thought.solved = self.ground_truth_evaluator(eval_state)
            except:
                new_thought.solved = False
            self.thoughts.append(new_thought)

        self.logger.info(
            "GroundTruth 操作 %d 评估了 %d 个思维，其中 %d 个解决了问题",
            self.id,
            len(self.thoughts),
            len([t for t in self.thoughts if t.solved]),
        )


class Selector(Operation):
    """
    选择器操作：从前驱思维中选择特定的思维。
    
    常用于将思维分组以执行不同的后续操作（如 GoT 中的分组处理）。
    """

    operation_type: OperationType = OperationType.selector

    def __init__(self, selector: Callable[[List[Thought]], List[Thought]]) -> None:
        """
        初始化 Selector 操作。

        :param selector: 选择函数，接收思维列表，返回选中的思维列表
        :type selector: Callable[[List[Thought]], List[Thought]]
        """
        super().__init__()
        self.selector: Callable[[List[Thought]], List[Thought]] = selector
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        返回选中的思维列表。

        :return: 选中的思维列表
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        执行 Selector 操作。
        
        如果没有前驱思维，使用 kwargs 构造一个初始思维。

        :param lm: 语言模型
        :param prompter: 提示词生成器
        :param parser: 响应解析器
        :param kwargs: 额外参数
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(kwargs)]

        self.thoughts = [
            Thought.from_thought(thought)
            for thought in self.selector(previous_thoughts)
        ]

        for thought in self.thoughts:
            self.logger.debug("选中思维 %d，状态: %s", thought.id, thought.state)

        self.logger.info("Selector 操作 %d 选中了 %d 个思维", self.id, len(self.thoughts))


class AdvanceSubquestion(Operation):
    """
    multiAgentGoT 专用：在同一个 Thought state 中推进到下一跳子问题。

    - 把当前 hop 的 refined partial_answer 写入 bindings（如 "#1": "<answer>"）
    - sub_id += 1，并切换 agent_role="retriever"、phase=1
    - 清空本跳临时字段（evidence/partial/candidates 等）
    """

    operation_type: OperationType = OperationType.advance_subquestion

    def __init__(self, max_hops: int) -> None:
        super().__init__()
        self.max_hops = max(1, int(max_hops))
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    @staticmethod
    def _as_str(x) -> str:
        return str(x).strip() if x is not None else ""

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thoughts()
        if not previous_thoughts:
            return

        self.thoughts = []
        for thought in previous_thoughts:
            state = dict(thought.state) if isinstance(thought.state, dict) else {}
            sub_id = int(state.get("sub_id", 0) or 0)
            subquestions = state.get("subquestions") or []
            if not isinstance(subquestions, list):
                subquestions = []

            # Record binding for current hop (1-indexed: #1 is hop0 answer).
            ans = self._as_str(state.get("partial_answer") or state.get("current") or "")
            if ans:
                bindings = state.get("bindings") or {}
                if not isinstance(bindings, dict):
                    bindings = {}
                bindings[f"#{sub_id + 1}"] = ans
                state["bindings"] = bindings

            next_sub_id = sub_id + 1
            if next_sub_id >= min(self.max_hops, len(subquestions) or self.max_hops):
                # No next hop; just carry state forward unchanged.
                self.thoughts.append(Thought(state))
                continue

            state["sub_id"] = next_sub_id
            state["subquestion"] = self._as_str(subquestions[next_sub_id]) if next_sub_id < len(subquestions) else ""
            state["agent_role"] = "retriever"
            state["phase"] = 1

            # Reset per-hop fields
            state["evidence_spans"] = []
            state["evidence_summary"] = ""
            state["pred_paragraph_support_idx"] = None
            state["partial_answer"] = ""
            state["current"] = ""
            state["confidence"] = 0.0
            state["candidate_answers"] = []
            state.pop("critique", None)
            state.pop("validation_decision", None)
            state.pop("reason_code", None)
            state.pop("suggested_action", None)

            self.thoughts.append(Thought(state))


class CriticVerifyAndBacktrack(Operation):
    """
    带有回溯机制的评估器操作：
    由 Critic 角色检验前驱思维（证据和局部结论）。
    如果检验不通过（REJECT），则抛出 BacktrackSignal 异常，打回给指定的起点重新生成。
    """

    operation_type: OperationType = OperationType.critic_verify_and_backtrack

    def __init__(
        self,
        target_backtrack_op: Operation,
        max_retries: int = 3,
        target_backtrack_reasoner_op: Operation | None = None,
    ) -> None:
        """
        初始化 CriticVerifyAndBacktrack 操作。

        :param target_backtrack_op: 如果检验失败，要回溯到的目标操作（通常是该分支的 Retriever）
        :type target_backtrack_op: Operation
        :param target_backtrack_reasoner_op: 如果“证据对但推理错”，可只回溯到 Reasoner（可选）
        :type target_backtrack_reasoner_op: Operation | None
        :param max_retries: 最大允许的回溯重试次数
        :type max_retries: int
        """
        super().__init__()
        self.target_backtrack_op: Operation = target_backtrack_op
        self.target_backtrack_reasoner_op: Operation | None = target_backtrack_reasoner_op
        self.max_retries: int = max_retries
        self.current_retries: int = 0
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
            self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        previous_thoughts: List[Thought] = self.get_previous_thoughts()
        
        if len(previous_thoughts) == 0:
            return

        for thought in previous_thoughts:
            base_state = thought.state
            
            # 强制设置为 critic 角色
            if hasattr(lm, "set_role"):
                lm.set_role("critic")
                
            # Avoid passing agent_role twice (it already exists in base_state for multiAgentGoT).
            prompt_kwargs = dict(base_state) if isinstance(base_state, dict) else {}
            prompt_kwargs.pop("agent_role", None)
            prompt = prompter.generate_prompt(1, agent_role="critic", **prompt_kwargs)
            self.logger.debug("Critic LLM 提示词: %s", prompt)
            
            responses = lm.get_response_texts(lm.query(prompt, num_responses=1))
            self.logger.debug("Critic LLM 响应: %s", responses)
            
            parsed_states = parser.parse_generate_answer(base_state, responses)
            if not parsed_states:
                continue
                
            parsed_state = parsed_states[0]
            
            # 检查 Critic 的验证决定
            if parsed_state.get("validation_decision") == "REJECT":
                if self.current_retries < self.max_retries:
                    self.current_retries += 1
                    reason_code = parsed_state.get("reason_code") or ""
                    time_facet = parsed_state.get("time_facet") or ""
                    suggested = (parsed_state.get("suggested_action") or "").strip().lower()
                    # Persist feedback for this hop across retries via controller's problem_parameters reference.
                    pp_ref = kwargs.get("__pp_ref")
                    if isinstance(pp_ref, dict):
                        fb_by_hop = pp_ref.get("retry_feedback_by_hop")
                        if not isinstance(fb_by_hop, dict):
                            fb_by_hop = {}
                            pp_ref["retry_feedback_by_hop"] = fb_by_hop
                        hop_key = parsed_state.get("sub_id", base_state.get("sub_id"))
                        try:
                            hop_key_int = int(hop_key)
                            hop_key_str = str(hop_key_int)
                        except Exception:
                            hop_key_int = None
                            hop_key_str = str(hop_key)
                        fb_by_hop[hop_key_str] = {
                            "critique": parsed_state.get("critique", ""),
                            "reason_code": reason_code,
                            "time_facet": time_facet,
                            "suggested_action": suggested,
                            "previous_partial_answer": base_state.get("partial_answer", ""),
                            "previous_evidence_summary": base_state.get("evidence_summary", ""),
                            "subquestion": base_state.get("subquestion", ""),
                        }
                    self.logger.warning(
                        "Critic 拒绝了当前结果，触发第 %d 次回溯。原因: %s (reason_code=%s)",
                        self.current_retries, parsed_state.get("critique", "未知")
                        , reason_code
                    )
                    target = self.target_backtrack_op
                    # If evidence looks OK but reasoning is wrong, only backtrack to reasoner if provided.
                    if suggested == "backtrack_reason" and self.target_backtrack_reasoner_op is not None:
                        target = self.target_backtrack_reasoner_op
                    # 抛出回溯信号，打回给目标节点
                    raise BacktrackSignal(
                        target,
                        reason=(
                            f"Critic REJECT (reason_code={reason_code}, time_facet={time_facet}, "
                            f"suggested_action={suggested})"
                        ),
                    )
                else:
                    self.logger.warning("达到最大重试次数 (%d)，强行通过。", self.max_retries)
            else:
                # On PASS, clear persisted feedback for this hop to avoid contaminating later steps.
                pp_ref = kwargs.get("__pp_ref")
                if isinstance(pp_ref, dict):
                    fb_by_hop = pp_ref.get("retry_feedback_by_hop")
                    if isinstance(fb_by_hop, dict):
                        hop_key = parsed_state.get("sub_id", base_state.get("sub_id"))
                        try:
                            hop_key_str = str(int(hop_key))
                        except Exception:
                            hop_key_str = str(hop_key)
                        fb_by_hop.pop(hop_key_str, None)
            
            # 如果 PASS 或达到最大重试次数，则保留该思维
            new_state = {**base_state, **parsed_state}
            self.thoughts.append(Thought(new_state))
            
        self.logger.info("CriticVerifyAndBacktrack 操作 %d 完成，保留了 %d 个思维", self.id, len(self.thoughts))
