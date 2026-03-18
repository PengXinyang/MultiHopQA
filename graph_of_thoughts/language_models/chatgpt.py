"""
ChatGPT 语言模型模块：实现 OpenAI GPT 系列模型的接口。
"""

import backoff  # 网络不稳定或 API 报错时，按指数退避策略自动重试
import os
import random
import time
from typing import List, Dict, Union
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    """
    ChatGPT 类，使用 OpenAI API 与 GPT 系列模型交互。
    
    继承自 AbstractLanguageModel，实现其抽象方法。
    支持 GPT-4、GPT-3.5-turbo 等模型。
    """

    def __init__(
            self, config_path: str = "", model_name: str = "chatgpt", cache: bool = False
    ) -> None:
        """
        初始化 ChatGPT 实例。

        :param config_path: 配置文件路径，默认为空（使用默认路径）
        :type config_path: str
        :param model_name: 模型名称，用于从配置中选择正确的配置项，默认为 'chatgpt'
        :type model_name: str
        :param cache: 是否缓存响应，默认为 False
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        
        # model_id: 实际使用的模型 ID，如 gpt-4, gpt-3.5-turbo 等
        self.model_id: str = self.config["model_id"]
        # 每 1000 个 token 的费用
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # temperature: 控制输出的随机性，值越高越随机
        self.temperature: float = self.config["temperature"]
        # max_tokens: 生成的最大 token 数
        self.max_tokens: int = self.config["max_tokens"]
        # stop: 停止序列，模型遇到此序列时停止生成
        self.stop: Union[str, List[str]] = self.config["stop"]
        # organization: OpenAI 组织 ID
        self.organization: str = self.config["organization"]
        if self.organization == "":
            self.logger.warning("OPENAI_ORGANIZATION 未设置")
        
        # API 密钥：优先从环境变量获取，否则从配置文件获取
        self.api_key: str = os.getenv("OPENAI_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("OPENAI_API_KEY 未设置")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=self.api_key, organization=self.organization)

    def query(
            self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        向 OpenAI 模型发送查询请求。

        :param query: 发送给模型的查询内容
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: OpenAI 模型的响应
        :rtype: Union[List[ChatCompletion], ChatCompletion]
        """
        # 检查缓存
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{"role": "user", "content": query}], num_responses)
        else:
            # 多响应请求：分批处理，带重试机制
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{"role": "user", "content": query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    # 失败时减半重试数量
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"ChatGPT 请求出错: {e}，使用 {next_try} 个样本重试"
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
        发送聊天消息到 OpenAI 模型并获取响应。
        使用指数退避策略处理 OpenAI 错误。

        :param messages: 消息列表，每个消息是包含 role 和 content 的字典
        :type messages: List[Dict]
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: OpenAI 模型的响应
        :rtype: ChatCompletion
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        # 更新 token 计数和费用
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
                self.prompt_token_cost * prompt_tokens_k
                + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"ChatGPT 响应: {response}"
            f"\n当前累计费用: ${self.cost:.4f}"
        )
        return response

    def get_response_texts(
            self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        从查询响应中提取文本内容。

        :param query_response: OpenAI 模型的响应（单个或列表）
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: 响应文本列表
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
