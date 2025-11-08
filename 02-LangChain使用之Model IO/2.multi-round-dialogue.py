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

chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)

sys_message = SystemMessage(
    content="You are a helpful assistant.",
)

human_message = HumanMessage(content="你好，请你介绍一下你自己")

messages = [sys_message, human_message]

# 调用大模型，传入messages
response = chat_model.invoke(messages)

print(response.content)

# 继续对话
response1 = chat_model.invoke("我之前的问题是什么？")

print(response1.content)
