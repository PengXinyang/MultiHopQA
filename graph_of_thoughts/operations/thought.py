"""
思维节点模块：表示 LLM 在推理过程中产生的一个"思维"状态。
"""

from __future__ import annotations
import logging
from typing import Iterator, Dict, Optional
import itertools


class Thought:
    """
    表示一个 LLM 思维节点，包含状态（由解析器构造）和各种标志位。
    
    属性：
        state: 思维的状态字典，存储当前推理的上下文和结果
        score: 思维的评分（越高越好或越低越好，取决于具体任务）
        valid: 思维是否有效（通过验证）
        solved: 思维是否正确解决了问题（与标准答案比较）
    """

    _ids: Iterator[int] = itertools.count(0)

    def __init__(self, state: Optional[Dict] = None) -> None:
        """
        初始化一个新的 Thought 实例。

        :param state: 思维的状态字典，默认为 None
        :type state: Optional[Dict]
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Thought._ids)
        self.state: Dict = state
        self._score: float = 0.0
        self._valid: bool = False
        self._solved: bool = False
        self.scored: bool = False          # 是否已评分
        self.validated: bool = False       # 是否已验证
        self.compared_to_ground_truth: bool = False  # 是否已与标准答案比较

    @staticmethod
    def from_thought(thought: Thought) -> Thought:
        """
        从现有思维创建一个新的思维副本。

        :param thought: 要克隆的思维实例
        :type thought: Thought
        :return: 复制了属性的新 Thought 实例
        :rtype: Thought
        """
        new_thought = Thought(thought.state)
        new_thought.score = thought.score
        new_thought.valid = thought.valid
        new_thought.solved = thought.solved
        new_thought.scored = thought.scored
        new_thought.validated = thought.validated
        new_thought.compared_to_ground_truth = thought.compared_to_ground_truth
        return new_thought

    @property
    def valid(self) -> bool:
        """
        获取思维的有效性。

        :return: 思维是否有效
        :rtype: bool
        """
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        """
        设置思维的有效性，同时将 validated 标志设为 True。

        :param valid: 思维是否有效
        :type valid: bool
        """
        self.validated = True
        self._valid = valid

    @property
    def score(self) -> float:
        """
        获取思维的评分。

        :return: 思维的评分
        :rtype: float
        """
        return self._score

    @score.setter
    def score(self, new_score: float) -> None:
        """
        设置思维的评分，同时将 scored 标志设为 True。

        :param new_score: 思维的评分
        :type new_score: float
        """
        self.scored = True
        self._score = new_score

    @property
    def solved(self) -> bool:
        """
        获取思维是否正确解决了问题。

        :return: 是否解决了问题
        :rtype: bool
        """
        return self._solved

    @solved.setter
    def solved(self, solved: bool) -> None:
        """
        设置思维是否解决了问题，同时将 compared_to_ground_truth 标志设为 True。

        :param solved: 是否解决了问题
        :type solved: bool
        """
        self.compared_to_ground_truth = True
        self._solved = solved
