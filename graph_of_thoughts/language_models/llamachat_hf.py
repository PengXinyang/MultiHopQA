"""
LLaMA 2 HuggingFace 语言模型模块：通过 HuggingFace Transformers 库使用 LLaMA 2 模型。

注意：此模块需要安装 transformers、torch 和 bitsandbytes 库，
并且需要足够的 GPU 内存来加载模型。
"""

import os
import torch
from typing import List, Dict, Union
from .abstract_language_model import AbstractLanguageModel


class Llama2HF(AbstractLanguageModel):
    """
    使用 HuggingFace 库调用 LLaMA 2 模型的接口类。
    
    支持 4-bit 量化以减少显存占用。
    """

    def __init__(
        self, config_path: str = "", model_name: str = "llama7b-hf", cache: bool = False
    ) -> None:
        """
        初始化 Llama2HF 实例。

        :param config_path: 配置文件路径，默认为空（使用默认路径）
        :type config_path: str
        :param model_name: LLaMA 模型变体名称，默认为 "llama7b-hf"
        :type model_name: str
        :param cache: 是否缓存响应，默认为 False
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        
        # model_id: 详细的模型标识
        self.model_id: str = self.config["model_id"]
        # 每 1000 个 token 的费用
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # temperature: 控制输出的随机性
        self.temperature: float = self.config["temperature"]
        # top_k: Top-K 采样参数
        self.top_k: int = self.config["top_k"]
        # max_tokens: 生成的最大 token 数
        self.max_tokens: int = self.config["max_tokens"]

        # 重要：必须在导入 transformers 之前设置缓存目录
        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]
        import transformers

        # 加载模型配置和 4-bit 量化配置
        hf_model_id = f"meta-llama/{self.model_id}"
        model_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 加载分词器和模型
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        torch.no_grad()

        # 创建文本生成 pipeline
        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-generation"
        )

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        向 LLaMA 2 模型发送查询请求。

        :param query: 发送给模型的查询内容
        :type query: str
        :param num_responses: 期望的响应数量，默认为 1
        :type num_responses: int
        :return: LLaMA 2 模型的响应列表
        :rtype: List[Dict]
        """
        # 检查缓存
        if self.cache and query in self.response_cache:
            return self.response_cache[query]
        
        sequences = []
        # 构造 LLaMA 2 的对话格式
        query = f"<s><<SYS>>You are a helpful assistant. Always follow the intstructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {query} [/INST]"
        
        for _ in range(num_responses):
            sequences.extend(
                self.generate_text(
                    query,
                    do_sample=True,
                    top_k=self.top_k,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_tokens,
                )
            )
        
        # 从生成的序列中提取响应文本（去除输入部分）
        response = [
            {"generated_text": sequence["generated_text"][len(query) :].strip()}
            for sequence in sequences
        ]
        
        # 缓存响应
        if self.cache:
            self.response_cache[query] = response
        return response

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        从查询响应中提取文本内容。

        :param query_responses: query 方法返回的响应字典列表
        :type query_responses: List[Dict]
        :return: 响应文本列表
        :rtype: List[str]
        """
        return [query_response["generated_text"] for query_response in query_responses]
