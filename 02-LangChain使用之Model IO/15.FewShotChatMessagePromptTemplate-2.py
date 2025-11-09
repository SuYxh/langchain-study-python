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


# 1. 定义示例集合 - 这是"剧本"的核心片段
# 每个示例都展示了用户如何提问，AI应该如何以"幽默程序员"的身份回答
examples = [
    {
        "input": "我的代码有bug，怎么办？",
        "output": "别慌，代码里的bug不是bug，是未公开的特性。让我看看你的'特性'是什么。"
    },
    {
        "input": "为什么我的程序这么慢？",
        "output": "因为它在深思熟虑，就像你面对一个复杂的需求一样。或者，可能只是你忘了写索引。"
    },
    {
        "input": "我想学Python，难吗？",
        "output": "不难，Python的语法就像说英语一样简单。唯一的问题是，计算机有时候比较'笨'，你得把话说得特别清楚。"
    }
]

# 2. 创建示例格式模板 - 定义如何将每个示例转换成"对话"
# 这里我们把用户的输入变成 HumanMessage，AI的回答变成 AIMessage
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 3. 创建 FewShotChatMessagePromptTemplate 实例
# 这会把我们的示例和格式组合成一个可重用的"剧本片段"
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)
print('few_shot_prompt', few_shot_prompt.format())
print('\n\n')


# 4. 构建最终的完整提示词
# 完整的提示词 = 系统角色设定 + 示例剧本 + 最终问题
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个幽默风趣的程序员助手，擅长用程序员梗来回答技术问题。"),
    few_shot_prompt,  # 插入我们创建的示例剧本
    ("human", "{input}"),  # 用户最终的问题
])
print('final_prompt', final_prompt.format(input="我的电脑又蓝屏了，是不是该换Mac了？"))
print('\n\n')


# 5. 创建链并调用
chain = final_prompt | chat_model

# 测试一下
response = chain.invoke({"input": "我的电脑又蓝屏了，是不是该换Mac了？"})
print(response.content)