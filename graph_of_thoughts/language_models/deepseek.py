"""
DeepSeek 语言模型模块：实现 DeepSeek 模型的接口。

DeepSeek 提供 OpenAI 兼容的 API，因此使用 OpenAI SDK 调用。
"""

import backoff
import os
import random
import time
from typing import Dict, List, Union

from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class DeepSeek(AbstractLanguageModel):
    """
    DeepSeek 模型适配类。

    行为尽量与 ChatGPT 类保持一致，区别在于：
    - 使用 DeepSeek 的 API Key（环境变量 DEEPSEEK_API_KEY 或配置中的 api_key）
    - 使用 DeepSeek 的 base_url（默认 https://api.deepseek.com，可在配置里覆盖）
    - model_id 默认为 DeepSeek 的模型名（例如 deepseek-chat）
    """

    def __init__(
        self,
        config_path: str = "",
        model_name: str = "deepseek",
        cache: bool = False,
    ) -> None:
        """
        初始化 DeepSeek 实例。

        :param config_path: 配置文件路径，默认为空（使用默认路径）
        :type config_path: str
        :param model_name: 模型名称，默认为 'deepseek'
        :type model_name: str
        :param cache: 是否缓存响应，默认为 False
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]

        # DeepSeek 模型信息与计费参数
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]

        # 采样与生成相关参数
        self.temperature: float = self.config["temperature"]
        self.max_tokens: int = self.config["max_tokens"]
        self.stop: Union[str, List[str]] = self.config["stop"]

        # DeepSeek 不需要 organization 字段，预留保持结构一致
        self.organization: str = self.config.get("organization", "")

        # API Key：优先从环境变量 DEEPSEEK_API_KEY 获取，否则从配置文件获取
        self.api_key: str = os.getenv("DEEPSEEK_API_KEY", self.config.get("api_key", ""))
        if self.api_key == "":
            raise ValueError("DEEPSEEK_API_KEY 未设置，且配置文件中未提供 api_key")

        # base_url：DeepSeek 的 OpenAI 兼容端点
        self.base_url: str = self.config.get("base_url", "https://api.deepseek.com")

        # 使用 OpenAI SDK，以自定义 base_url 调用 DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization or None,
        )

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        向 DeepSeek 模型发送查询请求。

        接口与 ChatGPT.query 保持一致：
        - 支持 num_responses 多样本
        - 返回单个或多个 ChatCompletion 对象

        :param query: 发送给模型的查询内容
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: DeepSeek 模型的响应
        :rtype: Union[List[ChatCompletion], ChatCompletion]
        """
        # 检查缓存
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], 1)
        else:
            # DeepSeek(OpenAI-compatible) 端点常见限制：仅支持 n=1。
            # 因此这里通过多次 n=1 的调用来“模拟”多样本，避免 400: Invalid n value。
            response = []
            total_num_attempts = num_responses
            remaining = num_responses
            while remaining > 0 and total_num_attempts > 0:
                try:
                    res = self.chat([{"role": "user", "content": query}], 1)
                    response.append(res)
                    remaining -= 1
                except Exception as e:
                    self.logger.warning(
                        f"DeepSeek 请求出错: {e}，稍后重试 (remaining={remaining})"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        # 缓存响应
        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> ChatCompletion:
        """
        发送多轮对话消息给 DeepSeek 模型。

        使用与 OpenAI 相同的聊天补全接口，只是通过 base_url 指向 DeepSeek。

        :param messages: 消息列表，每个消息是包含 role 和 content 的字典
        :type messages: List[Dict]
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: DeepSeek 模型的响应
        :rtype: ChatCompletion
        """
        # OpenAI-compatible 端点常见限制：n 仅支持 1；max_tokens 上限通常为 8192。
        safe_max_tokens = min(int(self.max_tokens), 8192)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=safe_max_tokens,
            n=1,
            stop=self.stop,
        )

        # 统计 token 用量与费用
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )

        self.logger.info(
            f"DeepSeek 响应: {response}"
            f"\n当前累计费用: ${self.cost:.4f}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        从 DeepSeek 的 ChatCompletion（或其列表）中提取纯文本回复。

        :param query_response: DeepSeek 模型的响应
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: 响应文本列表
        :rtype: List[str]
        """
        # typing.List is not valid for isinstance(); use built-in list.
        if not isinstance(query_response, list):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
