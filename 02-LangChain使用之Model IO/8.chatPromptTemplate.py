import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(
    model=model_name, base_url=base_url, api_key=api_key, streaming=True
)

# 定义聊天提示词模版
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的AI机器人，你的名字是{name}。"),
        ("human", "你好，最近怎么样？"),
        ("ai", "我很好，谢谢！"),
        ("human", "{user_input}"),
    ]
)


# 格式化聊天提示词模版中的变量
messages = chat_template.invoke(input={"name": "小明", "user_input": "你叫什么名字？"})
# 打印格式化后的聊天提示词模版内容
print("messages:", messages)
print("messages-type:", type(messages))


# 格式化聊天提示词模版中的变量，返回字符串
messages2 = chat_template.format(name="小明", user_input="你叫什么名字？")
# 打印格式化后的聊天提示词模版内容
print("messages2:", messages2)
print("messages2-type:", type(messages2))

# 格式化聊天提示词模版中的变量，返回消息列表-推荐使用改方式
messages3 = chat_template.format_messages(name="小明", user_input="你叫什么名字？")
# 打印格式化后的聊天提示词模版内容
print("messages3:", messages3)
print("messages3-type:", type(messages3))

# 格式化聊天提示词模版中的变量 和 invoke 方法返回的结果是一样的
messages4 = chat_template.format_prompt(name="小明", user_input="你叫什么名字？")
# 打印格式化后的聊天提示词模版内容
print("messages4:", messages4)
print("messages4-type:", type(messages4))


# 使用 BaseChatPromptTemplate（嵌套的 ChatPromptTemplate）
nested_prompt_template1 = ChatPromptTemplate.from_messages(
    [("system", "我是一个人工智能助手，我的名字叫{name}")]
)
nested_prompt_template2 = ChatPromptTemplate.from_messages(
    [("human", "很高兴认识你,我的问题是{question}")]
)
prompt_template = ChatPromptTemplate.from_messages(
    [nested_prompt_template1, nested_prompt_template2]
)
# 格式化聊天提示词模版中的变量，返回消息列表-推荐使用改方式
messages5 = prompt_template.format_messages(name="小智", question="你为什么这么帅？")
# 打印格式化后的聊天提示词模版内容
print("messages5:", messages5)
print("messages5-type:", type(messages5))


# prompt_template10 = PromptTemplate.from_template(
#     template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。"
# )
# prompt10 = prompt_template10.format(product="电脑", aspect1="性能", aspect2="电池")
# print("prompt-10:", prompt10)

# print(type(prompt10))

# res = chat_model.invoke(prompt10)  # 使用对话模型调用
# print(res.content)
