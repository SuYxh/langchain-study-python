"""大语言模型配置和初始化模块

提供统一的LLM模型创建和配置功能。
"""

import os
import ssl
import httpx
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# # 在项目根目录自动查找并加载 .env 文件
# load_dotenv()


def create_siliconflow_model(model_name: str = "Qwen/Qwen3-8B") -> object:
    """创建硅基流动平台的聊天模型

    Args:
        model_name: 模型名称，默认为"Qwen/Qwen3-8B"

    Returns:
        初始化好的聊天模型实例

    Raises:
        ValueError: 当API密钥未找到时
    """
    # 确保环境变量已加载
    load_dotenv()

    # 获取API密钥
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "SILICONFLOW_API_KEY not found in .env file or environment variables."
        )

    # 解决SSL证书验证问题
    # 创建一个不验证SSL证书的httpx客户端
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    custom_client = httpx.Client(verify=False, timeout=30.0)  # 禁用SSL验证

    # 创建并返回模型
    model = init_chat_model(
        model_name,
        model_provider="openai",
        base_url="https://api.siliconflow.cn/v1",
        api_key=api_key,
        http_client=custom_client,  # 使用自定义的httpx客户端
    )

    return model


def create_model(provider: str = "siliconflow", model_name: str = None) -> object:
    """通用模型创建函数

    Args:
        provider: 模型提供商，目前支持"siliconflow"
        model_name: 模型名称，如果为None则使用默认值

    Returns:
        初始化好的聊天模型实例

    Raises:
        ValueError: 当提供商不支持时
    """
    if provider == "siliconflow":
        default_model = "Qwen/Qwen3-8B"
        return create_siliconflow_model(model_name or default_model)
    else:
        raise ValueError(f"不支持的模型提供商: {provider}")


if __name__ == "__main__":
    """测试模块功能"""
    try:
        # 测试创建模型
        model = create_siliconflow_model()
        print("模型创建成功！")

        # 测试模型调用
        response = model.invoke("你好，请简单介绍一下自己。")
        print(f"模型回复: {response.content}")

    except Exception as e:
        print(f"错误: {e}")
