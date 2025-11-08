import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(
    model=model_name, base_url=base_url, api_key=api_key, streaming=True
)


messages1 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是机器学习"),
]
messages2 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是AIGC"),
]
messages3 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是大模型技术"),
]
messages = [messages1, messages2, messages3]

# 调用batch
response = chat_model.batch(messages)  # pyright: ignore[reportArgumentType]

# 打印每个消息的回复
for i, msg in enumerate(response):
    print(f"消息 {i+1} 的回复: {msg.content}")
