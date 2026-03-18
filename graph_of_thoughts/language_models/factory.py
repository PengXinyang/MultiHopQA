"""
语言模型工厂模块：根据模型名称自动选择并实例化合适的语言模型类。
"""

from __future__ import annotations

from graph_of_thoughts.language_models import ChatGPT, DeepSeek, Gemini


def build_language_model(
        config_path: str,
        model_name: str,
        cache: bool = True,
):
    """
    根据模型名自动匹配并实例化语言模型类。

    命名约定：
    - `*-gcli`：使用 GCLI 代理（GCLIGemini），模型名去掉 `-gcli` 后缀
      - 例如：`gemini-2.5-flash-gcli` -> GCLIGemini(model_name='gemini-2.5-flash-gcli')
    - `gemini-*`：使用原生 Gemini SDK（Gemini）
    - `deepseek-*`：使用 DeepSeek API（DeepSeek）
    - 其它：默认使用 OpenAI/ChatGPT 兼容接口（ChatGPT）

    :param config_path: 配置文件路径
    :type config_path: str
    :param model_name: 模型名称
    :type model_name: str
    :param cache: 是否缓存响应，默认为 True
    :type cache: bool
    :return: 语言模型实例
    :rtype: AbstractLanguageModel
    """
    normalized = (model_name or "").strip()
    lower = normalized.lower()

    if lower.endswith("-gcli"):
        print("使用 GCLI Gemini 代理模型")
        # 延迟导入，避免未安装/未使用时影响其它模型
        from graph_of_thoughts.language_models.gcli_gemini import GCLIGemini
        return GCLIGemini(
            config_path,
            model_name=normalized,
            cache=cache,
        )
    
    if lower.startswith("gemini-"):
        print("使用 Gemini 原生模型")
        return Gemini(
            config_path,
            model_name=normalized,
            cache=cache,
        )
    
    if lower.startswith("deepseek-"):
        print("使用 DeepSeek 模型")
        return DeepSeek(
            config_path,
            model_name=normalized,
            cache=cache,
        )

    print("使用 ChatGPT/OpenAI 兼容模型")
    return ChatGPT(
        config_path,
        model_name=normalized,
        cache=cache,
    )
