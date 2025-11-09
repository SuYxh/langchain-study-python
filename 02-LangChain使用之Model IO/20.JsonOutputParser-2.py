import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)

joke_query = "告诉我一个笑话。"
# 定义Json解析器
parser = JsonOutputParser()

# 返回一些指令或模板，这些指令告诉系统如何解析或格式化输出数据
format_instructions = parser.get_format_instructions()
print('format_instructions:',format_instructions)
print("\n")

# 定义提示词模版
# 注意，提示词模板中需要部分格式化解析器的格式要求format_instructions
prompt = PromptTemplate(
    template="回答用户的查询.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    # 预填充某些变量的值，避免重复传入
    partial_variables={"format_instructions": format_instructions},
)

print(prompt.format(query=joke_query))
print("\n")

# 5.使用LCEL语法组合一个简单的链
chain = prompt | chat_model | parser
# 6.执行链
output = chain.invoke({"query": "给我讲一个笑话"})
print(output)
