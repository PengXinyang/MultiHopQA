"""
语言模型包：提供多种语言模型的统一接口。

支持的模型：
- ChatGPT: OpenAI GPT 系列模型
- Gemini: Google Gemini 模型（原生 SDK）
- GCLIGemini: 通过 GCLI 代理访问的 Gemini 模型
- DeepSeek: DeepSeek 模型
- Llama2HF: LLaMA 2 模型（HuggingFace）

使用 build_language_model() 工厂函数可以根据模型名自动选择合适的类。
"""

from .abstract_language_model import AbstractLanguageModel
from .chatgpt import ChatGPT
from .llamachat_hf import Llama2HF
from .deepseek import DeepSeek
from .gemini import Gemini
from .gcli_gemini import GCLIGemini
from .rotating import RotatingLanguageModel, LightweightModelGroup, HeavyModelGroup
from .factory import build_language_model
