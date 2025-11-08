import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatMessagePromptTemplate,
    AIMessagePromptTemplate,
)
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

# 创建消息模板
system_template = "你是一个专家{role}"
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
print("system_message_prompt:", system_message_prompt)
print("system_message_prompt-type:", type(system_message_prompt))
print("\n\n")

human_template = "给我解释{concept}，用浅显易懂的语言"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
print("human_message_prompt:", human_message_prompt)
print("human_message_prompt-type:", type(human_message_prompt))
print("\n\n")

# 组合成聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

print("chat_prompt:", chat_prompt)
print("chat_prompt-type:", type(chat_prompt))
print("\n\n")


# 格式化提示
formatted_messages = chat_prompt.format_messages(role="物理学家", concept="相对论")
print("formatted_messages:", formatted_messages)
print("formatted_messages-type:", type(formatted_messages))
print("--------------------------------------------------------------")


prompt = "今天我们授课的内容是{subject}"

chat_message_prompt = ChatMessagePromptTemplate.from_template(
    role="teacher", template=prompt
)

resp = chat_message_prompt.format(subject="我爱北京天安门")

print("resp:", resp)
print("resp-type:", type(resp))

resp2 = chat_message_prompt.format_messages(subject="我爱北京天安门")
print("resp2:", resp2)
print("resp2-type:", type(resp2))

print("--------------------------------------------------------------")
print("\n\n")


# 使用 SystemMessagePromptTemplate
system_prompt = SystemMessagePromptTemplate.from_template("你是一个{role}.")
print("system_prompt:", system_prompt)
print("system_prompt-type:", type(system_prompt))
print("\n\n")

# 使用 HumanMessagePromptTemplate
human_prompt = HumanMessagePromptTemplate.from_template("{user_input}")
print("human_prompt:", human_prompt)
print("human_prompt-type:", type(human_prompt))
print("\n\n")


# 示例 2: 使用 BaseMessage（已实例化的消息）
system_msg = SystemMessage(content="你是一个AI工程师。")
human_msg = HumanMessage(content="你好！")

print("system_msg:", system_msg)
print("system_msg-type:", type(system_msg))
print("\n\n")
print("human_msg:", human_msg)
print("human_msg-type:", type(human_msg))
print("\n\n")


# 示例 3: 使用 BaseChatPromptTemplate（嵌套的 ChatPromptTemplate）
nested_prompt = ChatPromptTemplate.from_messages([("system", "嵌套提示词")])
prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,  # MessageLike (BaseMessagePromptTemplate)
        human_prompt,  # MessageLike (BaseMessagePromptTemplate)
        system_msg,  # MessageLike (BaseMessage)
        human_msg,  # MessageLike (BaseMessage)
        nested_prompt,  # MessageLike (BaseChatPromptTemplate)
    ]
)
print("prompt:", prompt)
print("prompt-type:", type(prompt))
print("\n\n")

formatted_messages = prompt.format_messages(
    role="人工智能专家", user_input="介绍一下大模型的应用场景"
)
print("formatted_messages:", formatted_messages)
print("formatted_messages-type:", type(formatted_messages))
print("--------------------------------------------------------------")
print("\n\n")


template4 = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "你是{product}的客服助手。你的名字叫{name}"
        ),
        HumanMessagePromptTemplate.from_template("hello 你好吗？"),
        AIMessagePromptTemplate.from_template("我很好 谢谢!"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

prompt = template4.format_messages(product="AGI课堂", name="Bob", query="你是谁")

# 调用聊天模型
response = chat_model.invoke(prompt)

print(response.content)