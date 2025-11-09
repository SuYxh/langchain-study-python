import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)

messages = [
    SystemMessage(content="将以下内容从英语翻译成中文"),
    HumanMessage(content="It's a nice day today"),
]

result = chat_model.invoke(messages)
print(result)
print(type(result))
print('\n')
token_usage = result.response_metadata.get('token_usage', {})
print(token_usage)
print('\n\n')

parser = StrOutputParser()

# 使用parser处理model返回的结果
response = parser.invoke(result)

print(response)
print(type(response))

