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

# 1、创建示例集合
# 定义示例集合，用于向模型展示如何处理特定任务
# 这是一个字典列表，每个字典包含输入和对应的期望输出
# 这些示例会作为"教学材料"帮助模型理解任务要求
# 示例的质量和数量直接影响模型的表现
examples = [
    {"input": "北京天气怎么样", "output": "北京市"},
    {"input": "南京下雨吗", "output": "南京市"},
    {"input": "武汉热吗", "output": "武汉市"},
]

# 2、创建PromptTemplate实例：  控制每个示例在最终提示词中的展示格式
# {input} 和 {output} 是占位符，会被 examples 中的实际值替换
# 可以自定义格式，如使用不同的分隔符或添加额外信息
example_prompt = PromptTemplate.from_template(
    template="Input: {input}\nOutput: {output}"
)

# 3、创建FewShotPromptTemplate实例
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    # 定义在所有示例之后、实际输入之前的提示文本。 这部分内容会拼接在所有示例的后面，通常包含对实际问题的引导或指令， {input} 占位符会被用户的实际输入替换，相当于告诉模型：“现在轮到你来回答这个问题了”
    suffix="Input: {input}\nOutput:",  # 要放在示例后面的提示模板字符串
    # 定义模板中需要从外部传入的变量列表， 告诉模板哪些变量需要从外部传入， 这些变量会在调用时通过 invoke() 方法传入， 必须包含 suffix 中使用的所有变量
    input_variables=["input"],  # 传入的变量
)
# 4、调用
prompt = prompt.invoke({"input": "长沙多少度"})
print("===Prompt===")
print(prompt.to_string())

print("===Response===")
response = chat_model.invoke(prompt)
print(response.content)
