"""
提示词生成器模块：定义生成 LLM 提示词的抽象接口。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class Prompter(ABC):
    """
    提示词生成器的抽象基类，定义所有提示词生成器的接口。
    
    提示词生成器负责根据当前思维状态构造发送给语言模型的提示词。
    不同的任务需要实现不同的 Prompter 子类。
    """

    @abstractmethod
    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        生成聚合操作的提示词。
        
        聚合操作将多个思维合并为一个。

        :param state_dicts: 要聚合的思维状态列表
        :type state_dicts: List[Dict]
        :param kwargs: 额外的关键字参数
        :return: 聚合提示词
        :rtype: str
        """
        pass

    @abstractmethod
    def improve_prompt(self, **kwargs) -> str:
        """
        生成改进操作的提示词。
        
        思维状态以 **kwargs 形式传入，允许子类明确指定所需参数。

        :param kwargs: 思维状态和额外参数
        :return: 改进提示词
        :rtype: str
        """
        pass

    @abstractmethod
    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        """
        生成生成操作的提示词。
        
        这是最常用的提示词类型，用于让 LLM 生成新的推理步骤。
        思维状态以 **kwargs 形式传入，允许子类明确指定所需参数。

        :param num_branches: 提示词应要求 LLM 生成的响应数量
        :type num_branches: int
        :param kwargs: 思维状态和额外参数
        :return: 生成提示词
        :rtype: str
        """
        pass

    @abstractmethod
    def validation_prompt(self, **kwargs) -> str:
        """
        生成验证操作的提示词。
        
        验证操作检查思维是否有效/合理。
        思维状态以 **kwargs 形式传入，允许子类明确指定所需参数。

        :param kwargs: 思维状态和额外参数
        :return: 验证提示词
        :rtype: str
        """
        pass

    @abstractmethod
    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        生成评分操作的提示词。
        
        评分操作让 LLM 对思维进行打分。

        :param state_dicts: 要评分的思维状态列表（如果多个，一起评分）
        :type state_dicts: List[Dict]
        :param kwargs: 额外的关键字参数
        :return: 评分提示词
        :rtype: str
        """
        pass
