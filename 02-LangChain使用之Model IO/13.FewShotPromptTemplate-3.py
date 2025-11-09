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

# 情感分析例子
examples = [
    {"句子": "今天天气真好！", "情感": "积极"},
    {"句子": "这个电影太无聊了", "情感": "消极"},
    {"句子": "考试没考好，很难过", "情感": "消极"},
]

example_prompt = PromptTemplate.from_template("句子: {句子}\n情感: {情感}")

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请判断以下句子的情感倾向（积极/消极）：",
    suffix="句子: {句子}\n情感:",
    input_variables=["句子"],
)

# 使用
final_prompt = prompt.invoke({"句子": "我中了大奖！"})
print("=== 生成的完整提示词 ===")
print(final_prompt.to_string())

print("\n=== AI的情感分析 ===")
response = chat_model.invoke(final_prompt)
print(response.content)
