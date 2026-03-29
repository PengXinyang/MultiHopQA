"""
控制器模块：管理操作图的执行流程，生成图推理状态（Graph Reasoning State）。
"""

import json
import logging
from typing import List, Callable, Dict, Any
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations import GraphOfOperations, Thought
from graph_of_thoughts.operations.operations import BacktrackSignal
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser


class Controller:
    """
    控制器类，用于管理操作图（Graph of Operations）的执行流程，生成图推理状态。
    涉及语言模型调用、图操作执行、提示词生成和响应解析。
    """

    def __init__(
        self,
        lm: AbstractLanguageModel,
        graph: GraphOfOperations,
        prompter: Prompter,
        parser: Parser,
        problem_parameters: dict,
        event_sink: Callable[[Dict[str, Any]], None] | None = None,
    ) -> None:
        """
        初始化 Controller 实例。

        :param lm: 语言模型实例（AbstractLanguageModel 的子类）
        :type lm: AbstractLanguageModel
        :param graph: 要执行的操作图
        :type graph: GraphOfOperations
        :param prompter: 提示词生成器，用于构造发送给 LLM 的提示词
        :type prompter: Prompter
        :param parser: 解析器，用于解析 LLM 的响应
        :type parser: Parser
        :param problem_parameters: 问题的初始参数/状态
        :type problem_parameters: dict
        """
        self.logger = logging.getLogger(self.__class__.__module__)
        self.lm = lm
        self.graph = graph
        self.prompter = prompter
        self.parser = parser
        self.problem_parameters = problem_parameters
        self.event_sink = event_sink
        self.run_executed = False

    def _emit(self, event: Dict[str, Any]) -> None:
        if not callable(self.event_sink):
            return
        try:
            self.event_sink(event)
        except Exception:
            self.logger.debug("事件发送失败（已忽略）", exc_info=True)

    @staticmethod
    def _build_node_payload(thought: Thought, op_id: int, op_type: str) -> Dict[str, Any]:
        state = thought.state if isinstance(thought.state, dict) else {}
        conclusion = (
            state.get("answer")
            or state.get("partial_answer")
            or state.get("current")
            or state.get("evidence_summary")
            or ""
        )
        thinking = []
        for key in ("_llm_raw_responses", "_score_raw_responses"):
            val = state.get(key)
            if isinstance(val, list):
                thinking.extend([str(x) for x in val if str(x).strip()])
            elif isinstance(val, str) and val.strip():
                thinking.append(val)
        return {
            "id": f"t_{thought.id}",
            "thought_id": thought.id,
            "label": f"{state.get('agent_role', op_type)}@hop{state.get('sub_id', -1)}",
            "role": state.get("agent_role", ""),
            "hop": state.get("sub_id", -1),
            "phase": state.get("phase", -1),
            "op_id": op_id,
            "op_type": op_type,
            "score": thought.score if thought.scored else None,
            "conclusion": str(conclusion),
            "thinking": "\n\n".join(thinking).strip(),
        }

    def _reset_descendants(self, operation) -> None:
        """递归重置目标操作及其所有后继操作的状态"""
        operation.executed = False
        if hasattr(operation, 'thoughts'):
            operation.thoughts = []
        for succ in operation.successors:
            if succ.executed:
                self._reset_descendants(succ)

    def run(self) -> None:
        """
        运行控制器，按拓扑顺序执行操作图中的所有操作。
        确保程序在执行前处于有效状态。

        :raises AssertionError: 如果操作图没有根节点
        :raises AssertionError: 如果某操作的后继不在操作图中
        """
        self.logger.debug("检查程序是否处于有效状态")
        assert self.graph.roots is not None, "操作图没有根节点"
        self.logger.debug("程序状态有效")

        # 初始化执行队列：所有可执行的操作（前驱已完成）
        execution_queue = [
            operation
            for operation in self.graph.operations
            if operation.can_be_executed()
        ]

        while len(execution_queue) > 0:
            current_operation = execution_queue.pop(0)
            self.logger.info("正在执行操作 %s", current_operation.operation_type)
            op_type = current_operation.operation_type.name
            self._emit(
                {
                    "type": "op_start",
                    "op_id": current_operation.id,
                    "op_type": op_type,
                }
            )

            try:
                previous_thoughts = current_operation.get_previous_thoughts()
                before_ids = {t.id for t in current_operation.get_thoughts()}
                current_operation.execute(
                    # Pass a stable reference so operations can persist feedback across retries.
                    self.lm,
                    self.prompter,
                    self.parser,
                    __pp_ref=self.problem_parameters,
                    **self.problem_parameters,
                )
                self.logger.info("操作 %s 执行完成", current_operation.operation_type)
                new_thoughts = [
                    t for t in current_operation.get_thoughts() if t.id not in before_ids
                ]
                parent_ids = [f"t_{t.id}" for t in previous_thoughts]
                for thought in new_thoughts:
                    self._emit(
                        {
                            "type": "thought_created",
                            "op_id": current_operation.id,
                            "op_type": op_type,
                            "node": self._build_node_payload(thought, current_operation.id, op_type),
                            "parent_ids": parent_ids,
                        }
                    )

                # 检查后继操作是否可以执行
                for operation in current_operation.successors:
                    assert (
                        operation in self.graph.operations
                    ), "操作的后继不在操作图中"
                    if operation.can_be_executed() and operation not in execution_queue:
                        execution_queue.append(operation)

            except BacktrackSignal as backtrack:
                self.logger.warning("触发全局回溯！原因: %s", backtrack.reason)
                target_op = backtrack.target_operation
                self._emit(
                    {
                        "type": "backtrack",
                        "op_id": current_operation.id,
                        "op_type": op_type,
                        "target_op_id": target_op.id,
                        "reason": backtrack.reason,
                    }
                )
                
                # 1. 重置目标节点及其所有子节点的状态
                self._reset_descendants(target_op)
                
                # 2. 重新计算当前可执行队列
                execution_queue = [
                    op for op in self.graph.operations
                    if op.can_be_executed() and not op.executed
                ]
            finally:
                # 回溯导致 execute 抛错时也必须发 op_end，否则前端节点会一直处于「思考中」且不易排查
                self._emit(
                    {
                        "type": "op_end",
                        "op_id": current_operation.id,
                        "op_type": op_type,
                    }
                )

        self.logger.info("所有操作执行完成")
        self._emit({"type": "run_end"})
        self.run_executed = True

    def get_final_thoughts(self) -> List[List[Thought]]:
        """
        获取所有操作执行完毕后的最终思维结果。

        :return: 操作图中每个叶子节点的思维列表
        :rtype: List[List[Thought]]
        :raises AssertionError: 如果 run() 方法尚未执行
        """
        assert self.run_executed, "run() 方法尚未执行"
        return [operation.get_thoughts() for operation in self.graph.leaves]

    def output_graph(self, path: str) -> None:
        """
        将操作图的状态和结果序列化为 JSON 文件。

        :param path: 输出文件路径
        :type path: str
        """
        output = []
        for operation in self.graph.operations:
            operation_serialized = {
                "operation": operation.operation_type.name,
                "thoughts": [thought.state for thought in operation.get_thoughts()],
            }
            # 如果有评分，添加评分信息
            if any([thought.scored for thought in operation.get_thoughts()]):
                operation_serialized["scored"] = [
                    thought.scored for thought in operation.get_thoughts()
                ]
                operation_serialized["scores"] = [
                    thought.score for thought in operation.get_thoughts()
                ]
            # 如果有验证，添加验证信息
            if any([thought.validated for thought in operation.get_thoughts()]):
                operation_serialized["validated"] = [
                    thought.validated for thought in operation.get_thoughts()
                ]
                operation_serialized["validity"] = [
                    thought.valid for thought in operation.get_thoughts()
                ]
            # 如果与标准答案比较过，添加比较结果
            if any(
                [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
            ):
                operation_serialized["compared_to_ground_truth"] = [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
                operation_serialized["problem_solved"] = [
                    thought.solved for thought in operation.get_thoughts()
                ]
            output.append(operation_serialized)

        # 添加 token 使用量和费用信息
        output.append(
            {
                "prompt_tokens": self.lm.prompt_tokens,
                "completion_tokens": self.lm.completion_tokens,
                "cost": self.lm.cost,
            }
        )

        with open(path, "w") as file:
            file.write(json.dumps(output, indent=2))
