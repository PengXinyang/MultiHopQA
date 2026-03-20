"""
GoO 构建器：根据 AI 生成的 GoO 建议（JSON）构造可执行的 GraphOfOperations。

典型输入是 auto_goo_designer 保存的结果中的 `parsed` 字段，例如：
{
  "nodes": [{"id": "n1", "type": "Generate"}, ...],
  "edges": [{"from": "n1", "to": "n2"}, ...],
  "final_nodes": ["n_last"]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from graph_of_thoughts import operations


@dataclass
class BuiltGoO:
    """
    构建结果容器。

    attributes:
        graph: 可执行的 GraphOfOperations
        node_operations: 节点 id -> Operation 实例映射
        final_operations: final_nodes 对应的 Operation 列表
    """

    graph: operations.GraphOfOperations
    node_operations: Dict[str, operations.Operation]
    final_operations: List[operations.Operation]


def topologicalSort(node_ids: List[str], edges: List[Dict[str, str]]) -> List[str]:
    """
    对节点做拓扑排序，确保先添加前驱，再添加后继。
    """
    indegree = {nid: 0 for nid in node_ids}
    succ: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for e in edges:
        u = e["from"]
        v = e["to"]
        if u not in indegree or v not in indegree:
            continue
        succ[u].append(v)
        indegree[v] += 1

    queue = [nid for nid in node_ids if indegree[nid] == 0]
    order: List[str] = []

    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in succ[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(order) != len(node_ids):
        cycle_nodes, cycle_edges = findCycleDetails(node_ids, edges)
        raise ValueError(
            "GoO contains a cycle; cannot topologically sort. "
            f"cycle_nodes={cycle_nodes}, cycle_edges={cycle_edges}"
        )
    return order


def findCycleDetails(node_ids: List[str], edges: List[Dict[str, str]]) -> (List[str], List[List[str]]):
    """
    返回一个可读的环信息（节点序列和边序列）。

    返回格式：
    - cycle_nodes: 例如 ["A", "B", "C", "A"]
    - cycle_edges: 例如 [["A", "B"], ["B", "C"], ["C", "A"]]
    """
    graph: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for e in edges:
        u = e.get("from")
        v = e.get("to")
        if u in graph and v in graph:
            graph[u].append(v)

    visited: Dict[str, int] = {nid: 0 for nid in node_ids}  # 0=未访问, 1=访问中, 2=已完成
    stack: List[str] = []

    def dfs(u: str) -> Optional[List[str]]:
        visited[u] = 1
        stack.append(u)
        for v in graph[u]:
            if visited[v] == 0:
                found = dfs(v)
                if found:
                    return found
            elif visited[v] == 1:
                # 找到回边：从栈里截取环路径，并补上起点闭环
                idx = stack.index(v)
                return stack[idx:] + [v]
        stack.pop()
        visited[u] = 2
        return None

    cycle_nodes: List[str] = []
    for nid in node_ids:
        if visited[nid] == 0:
            found = dfs(nid)
            if found:
                cycle_nodes = found
                break

    if not cycle_nodes:
        # 兜底：理论上不会走到这里（已有环判定）
        return [], []

    cycle_edges: List[List[str]] = []
    for i in range(len(cycle_nodes) - 1):
        cycle_edges.append([cycle_nodes[i], cycle_nodes[i + 1]])
    return cycle_nodes, cycle_edges


def buildSelectorFromParams(params: Dict[str, Any]) -> Callable[[List[operations.Thought]], List[operations.Thought]]:
    """
    根据节点参数构造 Selector 函数。

    支持两种模式：
    - identity（默认）：原样返回
    - part_equals：按 thought.state[state_key] == equals 过滤
    """
    mode = (params or {}).get("mode", "identity")
    if mode == "part_equals":
        state_key = (params or {}).get("state_key", "part")
        expected = (params or {}).get("equals", "")

        def _selector(thoughts: List[operations.Thought]) -> List[operations.Thought]:
            return [t for t in thoughts if t.state.get(state_key) == expected]

        return _selector

    return lambda thoughts: thoughts


def buildGraphFromGooDesign(
    design: Dict[str, Any],
    scoring_function: Optional[Callable[[Any], Any]] = None,
    validate_function: Optional[Callable[[Dict], bool]] = None,
    ground_truth_evaluator: Optional[Callable[[Dict], bool]] = None,
    selector_registry: Optional[Dict[str, Callable[[List[operations.Thought]], List[operations.Thought]]]] = None,
) -> BuiltGoO:
    """
    从 GoO 设计字典构造 GraphOfOperations。

    :param design: GoO 设计字典（通常是 auto_goo_* 文件中的 parsed）
    :param scoring_function: Score 节点使用的评分函数（可选）
    :param validate_function: ValidateAndImprove 节点使用的验证函数（可选）
    :param ground_truth_evaluator: GroundTruth 节点使用的标准答案评估函数（可选）
    :param selector_registry: Selector 节点 id -> 自定义选择函数映射（可选）
    :return: BuiltGoO（包含 graph、节点映射和 final 操作）
    """
    nodes = design.get("nodes", [])
    edges = design.get("edges", [])
    final_node_ids = design.get("final_nodes", [])

    if not nodes:
        raise ValueError("GoO design has no nodes.")

    node_ids = [n["id"] for n in nodes]
    node_by_id = {n["id"]: n for n in nodes}
    order = topologicalSort(node_ids, edges)

    node_operations: Dict[str, operations.Operation] = {}

    # 1) 创建所有 Operation 实例
    for nid in order:
        node = node_by_id[nid]
        op_type = str(node.get("type", "")).strip()
        params = node.get("params", {}) or {}
        instruction = node.get("instruction", "")

        if op_type == "Generate":
            op = operations.Generate(
                int(params.get("num_branches_prompt", 1)),
                int(params.get("num_branches_response", 1)),
            )
        elif op_type == "Score":
            op = operations.Score(
                int(params.get("num_samples", 1)),
                bool(params.get("combined_scoring", False)),
                scoring_function,
            )
        elif op_type == "KeepBestN":
            op = operations.KeepBestN(
                int(params.get("n", 1)),
                bool(params.get("higher_is_better", True)),
            )
        elif op_type == "Aggregate":
            op = operations.Aggregate(int(params.get("num_responses", 1)))
        elif op_type == "ValidateAndImprove":
            op = operations.ValidateAndImprove(
                int(params.get("num_samples", 1)),
                bool(params.get("improve", True)),
                int(params.get("num_tries", 3)),
                validate_function,
            )
        elif op_type == "Improve":
            op = operations.Improve()
        elif op_type == "KeepValid":
            op = operations.KeepValid()
        elif op_type == "GroundTruth":
            evaluator = ground_truth_evaluator or (lambda _state: False)
            op = operations.GroundTruth(evaluator)
        elif op_type == "Selector":
            if selector_registry and nid in selector_registry:
                selector_fn = selector_registry[nid]
            else:
                selector_fn = buildSelectorFromParams(params)
            op = operations.Selector(selector_fn)
        else:
            raise ValueError(f"Unsupported operation type in GoO design: {op_type}")

        # 将节点级 instruction 挂在 Operation 实例上，供 downstream 使用
        if instruction:
            setattr(op, "node_instruction", instruction)

        node_operations[nid] = op

    # 2) 连接边（设置前驱/后继关系）
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm not in node_operations or to not in node_operations:
            raise ValueError(f"Invalid edge {e}, node not found in nodes.")
        node_operations[to].add_predecessor(node_operations[frm])

    # 3) 按拓扑顺序加入 GraphOfOperations
    graph = operations.GraphOfOperations()
    for nid in order:
        graph.add_operation(node_operations[nid])

    final_operations: List[operations.Operation] = [
        node_operations[nid] for nid in final_node_ids if nid in node_operations
    ]

    return BuiltGoO(
        graph=graph,
        node_operations=node_operations,
        final_operations=final_operations,
    )


def loadGooDesignFromFile(path: str) -> Dict[str, Any]:
    """
    从文件读取 GoO 设计。

    兼容两种格式：
    - 纯 design JSON（直接包含 nodes/edges/final_nodes）
    - auto_goo 输出格式（包含 raw_text/parsed/prompt），优先返回 parsed
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "parsed" in obj and isinstance(obj["parsed"], dict):
        return obj["parsed"]
    return obj

