"""
操作图模块：定义操作图（Graph of Operations）数据结构。
操作图描述了思维操作的执行计划。
"""

from __future__ import annotations
from typing import List

from graph_of_thoughts.operations.operations import Operation


class GraphOfOperations:
    """
    操作图类，描述思维操作的执行计划。
    
    操作图是一个有向无环图（DAG），其中：
    - roots（根节点）：图的入口点，没有前驱操作
    - leaves（叶节点）：图的出口点，没有后继操作
    - operations：图中所有操作的列表
    """

    def __init__(self) -> None:
        """
        初始化一个新的操作图实例。
        初始状态下，操作列表、根节点和叶节点均为空。
        """
        self.operations: List[Operation] = []  # 所有操作
        self.roots: List[Operation] = []       # 根节点（入口）
        self.leaves: List[Operation] = []      # 叶节点（出口）

    def append_operation(self, operation: Operation) -> None:
        """
        将操作追加到图的所有叶节点之后。
        
        这是构建线性操作链的便捷方法。新操作会成为所有当前叶节点的后继，
        然后自身成为新的唯一叶节点。

        :param operation: 要追加的操作
        :type operation: Operation
        """
        self.operations.append(operation)

        if len(self.roots) == 0:
            # 第一个操作，同时是根节点
            self.roots = [operation]
        else:
            # 将新操作设为所有叶节点的后继
            for leaf in self.leaves:
                leaf.add_successor(operation)

        # 新操作成为唯一的叶节点
        self.leaves = [operation]

    def add_operation(self, operation: Operation) -> None:
        """
        将操作添加到图中，根据其前驱和后继关系更新根节点和叶节点。
        
        与 append_operation 不同，此方法允许更灵活的图结构，
        操作的前驱/后继需要在调用前通过 add_predecessor/add_successor 设置。

        :param operation: 要添加的操作
        :type operation: Operation
        :raises AssertionError: 如果第一个操作有前驱
        """
        self.operations.append(operation)
        
        if len(self.roots) == 0:
            # 第一个操作，必须没有前驱
            self.roots = [operation]
            self.leaves = [operation]
            assert (
                len(operation.predecessors) == 0
            ), "第一个操作不应有前驱"
        else:
            # 如果操作没有前驱，它是一个新的根节点
            if len(operation.predecessors) == 0:
                self.roots.append(operation)
            
            # 更新叶节点：如果某个前驱是叶节点，将其从叶节点列表移除
            for predecessor in operation.predecessors:
                if predecessor in self.leaves:
                    self.leaves.remove(predecessor)
            
            # 如果操作没有后继，它是一个叶节点
            if len(operation.successors) == 0:
                self.leaves.append(operation)
