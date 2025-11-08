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


sys_message = SystemMessage(
    content="You are a helpful assistant.",
)

human_message = HumanMessage(
    content="如果人工智能有感情，它们会如何表达对程序员的感谢？"
)
human_message1 = HumanMessage(
    content="如果让你设计一个完美的编程语言，它会有哪些创新特性？"
)

messages = [sys_message, human_message, human_message1]

stream_response = chat_model.stream(messages)

# 流式调用LLM获取响应
print("开始流式输出：")
for chunk in stream_response:
    # 逐个打印内容块
    print(
        chunk.content, end="", flush=True
    )  # 刷新缓冲区 (无换行符，缓冲区未刷新，内容可能不会立即显示)

print("\n流式输出结束")
