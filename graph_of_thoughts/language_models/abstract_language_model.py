"""
抽象语言模型模块：定义所有语言模型的基类接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any
import json
import os
import logging


class AbstractLanguageModel(ABC):
    """
    语言模型的抽象基类，定义所有语言模型实现必须遵循的接口。
    
    属性：
        config: 模型配置字典
        model_name: 模型名称
        cache: 是否缓存响应
        prompt_tokens: 累计消耗的提示词 token 数
        completion_tokens: 累计消耗的生成 token 数
        cost: 累计费用（美元）
    """

    def __init__(
        self, config_path: str = "", model_name: str = "", cache: bool = False
    ) -> None:
        """
        初始化抽象语言模型实例。

        :param config_path: 配置文件路径，默认为空字符串（使用默认路径）
        :type config_path: str
        :param model_name: 模型名称，默认为空字符串
        :type model_name: str
        :param cache: 是否缓存响应，默认为 False
        :type cache: bool
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict = None
        self.model_name: str = model_name
        self.cache = cache
        if self.cache:
            self.response_cache: Dict[str, List[Any]] = {}
        self.load_config(config_path)
        self.prompt_tokens: int = 0       # 累计提示词 token 数
        self.completion_tokens: int = 0   # 累计生成 token 数
        self.cost: float = 0.0            # 累计费用

    def load_config(self, path: str) -> None:
        """
        从指定路径加载配置文件。

        :param path: 配置文件路径。如果为空，默认使用当前目录下的 config.json
        :type path: str
        """
        if path == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "config.json")

        with open(path, "r") as f:
            self.config = json.load(f)

        self.logger.debug(f"已从 {path} 加载配置，模型: {self.model_name}")

    def clear_cache(self) -> None:
        """
        清空响应缓存。
        """
        self.response_cache.clear()

    @abstractmethod
    def query(self, query: str, num_responses: int = 1) -> Any:
        """
        向语言模型发送查询请求（抽象方法）。

        :param query: 发送给语言模型的查询内容
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: 语言模型的响应
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_response_texts(self, query_responses: Union[List[Any], Any]) -> List[str]:
        """
        从语言模型的响应中提取文本内容（抽象方法）。

        :param query_responses: 语言模型返回的响应对象
        :type query_responses: Union[List[Any], Any]
        :return: 响应文本列表
        :rtype: List[str]
        """
        pass
