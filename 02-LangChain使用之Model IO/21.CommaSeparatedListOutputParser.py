import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "你是一个靠谱的{role}"), ("human", "{question}")]
)


output_parser = CommaSeparatedListOutputParser()

# 返回一些指令或模板，这些指令告诉系统如何解析或格式化输出数据
format_instructions = output_parser.get_format_instructions()
print("format_instructions:", format_instructions)
print("\n")

messages = "大象,猩猩,狮子"
result = output_parser.parse(messages)
print("result:", result)
print(type(result))
print("\n")


prompt = PromptTemplate(
    template="生成5个关于{text}的列表，使用中文回复 .\n\n{format_instructions}",
    input_variables=["text"],
    # 预填充某些变量的值，避免重复传入
    partial_variables={"format_instructions": format_instructions},
)

print(prompt.format(text="动物"))
print("\n")

# 5.使用LCEL语法组合一个简单的链
chain = prompt | chat_model | output_parser
# 6.执行链
output = chain.invoke({"text": "动物"})
print(output)