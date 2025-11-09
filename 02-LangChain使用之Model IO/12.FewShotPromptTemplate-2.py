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
# 1. 创建示例集合 - 就像给AI看"标准答案"
examples = [
    {"食材": "土豆, 鸡蛋", "推荐菜": "土豆炒鸡蛋"},
    {"食材": "西红柿, 鸡蛋", "推荐菜": "西红柿炒鸡蛋"},
    {"食材": "白菜, 豆腐", "推荐菜": "白菜炖豆腐"},
]

# 2. 创建示例格式模板 - 定义"标准答案"的展示格式
example_prompt = PromptTemplate.from_template(template="食材: {食材}\n推荐菜: {推荐菜}")

# 3. 创建FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=examples,  # 教学示例
    example_prompt=example_prompt,  # 示例格式
    prefix="我是一个智能菜谱推荐助手。根据你提供的食材，我会推荐合适的菜名。\n以下是一些示例：",  # 开头说明
    suffix="食材: {食材}\n推荐菜:",  # 最后的问题格式
    input_variables=["食材"],  # 需要用户提供的变量
)

# 4. 使用示例
print("=== 生成的完整提示词 ===")
final_prompt = prompt.invoke({"食材": "黄瓜, 鸡蛋"})
print(final_prompt.to_string())

print("\n=== AI的推荐 ===")
response = chat_model.invoke(final_prompt)
print(response.content)
