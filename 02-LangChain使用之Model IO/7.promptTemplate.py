import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(
    model=model_name, base_url=base_url, api_key=api_key, streaming=True
)


# 基础模板 ------------------------------------------------
# 定义模板
prompt_template = PromptTemplate.from_template("请用简明的语言介绍一下{topic}。")

# 填充变量
prompt = prompt_template.format(topic="人工智能")
print("prompt-1:", prompt)
# 输出：请用简明的语言介绍一下人工智能。


# 多变量模板 ------------------------------------------------
prompt_template2 = PromptTemplate.from_template(
    "请用简明的语言介绍一下{topic}，并解释它的{aspect}。"
)
prompt2 = prompt_template2.format(topic="机器学习", aspect="应用")
print("prompt-2:", prompt2)
# 输出：请用简明的语言介绍一下机器学习，并解释它的应用。


# 嵌套模板 ------------------------------------------------
base_template = "请用简明的语言介绍一下{topic}。"
aspect_template = base_template + " 并解释它的{aspect}。"

prompt_template3 = PromptTemplate.from_template(aspect_template)
prompt3 = prompt_template3.format(topic="深度学习", aspect="基本原理")
print("prompt-3:", prompt3)
# 输出：请用简明的语言介绍一下深度学习。 并解释它的基本原理。


# 动态变量 ------------------------------------------------
def generate_topic():
    return "自然语言处理"


prompt_template4 = PromptTemplate.from_template("请介绍一下{topic}。")
prompt4 = prompt_template4.format(topic=generate_topic())
print("prompt-4:", prompt4)
# 输出：请介绍一下自然语言处理。


# 实例化过程中使用partial_variables变量 ------------------------------------------------
template5 = PromptTemplate(
    template="{foo}{bar}",
    input_variables=["foo", "bar"],
    partial_variables={"foo": "hello"},
)

prompt5 = template5.format(bar=" world")

print("prompt-5:", prompt5)

# 使用 PromptTemplate.partial() 方法创建部分提示模板 ------------------------------------------------
template6 = PromptTemplate(template="{foo}{bar}", input_variables=["foo", "bar"])
# 方式1：
partial_template6 = template6.partial(foo="hello")
prompt6 = partial_template6.format(bar="world")
print("prompt-6:", prompt6)


# 完整模板
full_template = """你是一个{role}，请用{style}风格回答：
问题：{question}
答案："""
# 预填充角色和风格
partial_template = PromptTemplate.from_template(full_template).partial(
    role="资深厨师", style="专业但幽默"
)
# 只需提供剩余变量
print("prompt-7:", partial_template.format(question="如何煎牛排？"))
print("prompt-7-2:", partial_template.format(style="专业", question="如何煎牛排？"))


# 部分填充变量
prompt_template8 = PromptTemplate.from_template(
    template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。",
    partial_variables={"aspect1": "电池", "aspect2": "屏幕"},
)
prompt8 = prompt_template8.format(product="笔记本电脑")
prompt8_2 = prompt_template8.invoke(input={"product": "笔记本电脑"})

print("prompt-8:", prompt8)
print("prompt-8-2:", prompt8_2)
print("prompt-8-3:", prompt8_2.to_string())
print("prompt-8-4:", prompt8_2.to_messages())


# 组合模板
template9 = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)
prompt9 = template9.format(topic="sports", language="spanish")
print("prompt-9:", prompt9)


prompt_template10 = PromptTemplate.from_template(
    template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。"
)
prompt10 = prompt_template10.format(product="电脑", aspect1="性能", aspect2="电池")
print("prompt-10:", prompt10)

print(type(prompt10))

res = chat_model.invoke(prompt10)  # 使用对话模型调用
print(res.content)
