"""
响应解析器模块：定义解析 LLM 响应的抽象接口。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Parser(ABC):
    """
    响应解析器的抽象基类，定义所有解析器的接口。
    
    解析器负责将语言模型返回的文本响应解析为结构化的思维状态。
    不同的任务需要实现不同的 Parser 子类。
    """

    @abstractmethod
    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        解析聚合操作的 LLM 响应。

        :param states: 用于生成提示词的思维状态列表
        :type states: List[Dict]
        :param texts: LLM 的响应文本列表
        :type texts: List[str]
        :return: 解析后的新思维状态（单个或列表）
        :rtype: Union[Dict, List[Dict]]
        """
        pass

    @abstractmethod
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        解析改进操作的 LLM 响应。

        :param state: 用于生成提示词的思维状态
        :type state: Dict
        :param texts: LLM 的响应文本列表
        :type texts: List[str]
        :return: 解析后的状态更新字典
        :rtype: Dict
        """
        pass

    @abstractmethod
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        解析生成操作的 LLM 响应。

        :param state: 用于生成提示词的思维状态
        :type state: Dict
        :param texts: LLM 的响应文本列表
        :type texts: List[str]
        :return: 解析后的新思维状态列表
        :rtype: List[Dict]
        """
        pass

    @abstractmethod
    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        解析验证操作的 LLM 响应。

        :param state: 用于生成提示词的思维状态
        :type state: Dict
        :param texts: LLM 的响应文本列表
        :type texts: List[str]
        :return: 思维是否有效
        :rtype: bool
        """
        pass

    @abstractmethod
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        解析评分操作的 LLM 响应。

        :param states: 用于生成提示词的思维状态列表
        :type states: List[Dict]
        :param texts: LLM 的响应文本列表
        :type texts: List[str]
        :return: 每个思维的评分列表
        :rtype: List[float]
        """
        pass
