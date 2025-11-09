# 导入相关包
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
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


# 定义示例组
examples = [
    {"input": "2🦜2", "output": "4"},
    {"input": "2🦜3", "output": "8"},
]

# 定义示例的消息格式提示词模板
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input} 是多少?"), ("ai", "{output}")]
)

# 定义FewShotChatMessagePromptTemplate对象
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples, example_prompt=example_prompt  # 示例组  # 示例提示词模板
)
print("few_shot_prompt", few_shot_prompt)
print("\n\n")


# 输出完整提示词的消息模板
final_prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一个数学奇才"), few_shot_prompt, ("human", "{input}")]
)
print("final_prompt", final_prompt)
print("\n\n")


resp = chat_model.invoke(
    final_prompt.invoke(input="2🦜4")  # pyright:ignore[reportArgumentType]
)
print(resp.content)
