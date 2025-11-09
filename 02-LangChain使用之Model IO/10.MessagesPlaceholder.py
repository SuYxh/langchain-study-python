import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import SecretStr

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(
    model=model_name, base_url=base_url, api_key=api_key, streaming=True
)

# 使用场景：多轮对话系统存储历史消息以及Agent的中间步骤处理此功能非常有用
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), MessagesPlaceholder("msgs")]
)
# prompt_template.invoke({"msgs": [HumanMessage(content="hi!")]})
prompt = prompt_template.format_messages(msgs=[HumanMessage(content="hi!")])
print("prompt:", prompt)
print("prompt-type:", type(prompt))

print("--------------------------------------------------------------")
print("\n\n")


prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
print("prompt2:", prompt2)
print("prompt2-type:", type(prompt2))
print("\n\n")


prompt2_messages = prompt2.format_messages(
    history=[HumanMessage(content="1+2*3 = ?"), AIMessage(content="1+2*3=7")],
    question="我刚才问题是什么？",
)
print("prompt2_messages:", prompt2_messages)
print("prompt2_messages-type:", type(prompt2_messages))

print("--------------------------------------------------------------")
print("\n\n")


# 定义HumanMessage对象
human_message = HumanMessage(content="学习编程的最好方法是什么？")
# 3.定义AIMessage对象
ai_message = AIMessage(
    content="""
1. 选择一门编程语言：选择一门你想学习的编程语言.
2.从基础开始：熟悉基本的编程概念，如变量、数据类型和控制结构.
3. 练习，练习，再练习：学习编程的最好方法是通过实践经验|
"""
)

# 4. 定义提示词
human_prompt = "用{word_count}个词总结我们到目前为止的对话"

# 5. 定义提示词模板
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

# 6.格式化聊天消息提示词模板
messages1 = chat_prompt.format_messages(
    conversation=[human_message, ai_message], word_count="10"
)
print("messages1:", messages1)
print("messages1-type:", type(messages1))
