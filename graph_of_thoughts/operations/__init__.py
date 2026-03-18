"""
操作模块包：定义 Graph of Thoughts 中的各种操作类型。

主要组件：
- Thought: 思维节点，表示推理过程中的一个状态
- GraphOfOperations: 操作图，定义操作的执行顺序
- Operation: 操作基类
- Generate: 生成新思维
- Score: 对思维评分
- KeepBestN: 保留最佳 N 个思维
- Aggregate: 聚合多个思维
- ValidateAndImprove: 验证并改进思维
- Improve: 改进思维
- KeepValid: 保留有效思维
- GroundTruth: 与标准答案比较
- Selector: 选择特定思维
"""

from .thought import Thought
from .graph_of_operations import GraphOfOperations
from .operations import (
    Operation,
    Score,
    ValidateAndImprove,
    Generate,
    Aggregate,
    KeepBestN,
    KeepValid,
    Selector,
    GroundTruth,
    Improve,
)
