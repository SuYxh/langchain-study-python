import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

dotenv.load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [("system", "你是一个靠谱的{role}"), ("human", "{question}")]
)
parser = JsonOutputParser()
# 方式1：
result = chat_model.invoke(
    chat_prompt_template.format_messages(
        role="人工智能专家",
        question="人工智能用英文怎么说？问题用q表示，答案用a表示，返回一个JSON格式",
    )
)
print(result)
print(type(result))
print("\n\n")

res = parser.invoke(result)
print(res)
print(type(res))

# 方式2：
# chain = chat_prompt_template | chat_model | parser
# chain.invoke({"role":"人工智能专家","question" : "人工智能用英文怎么说？问题用q表示，答案用a表示，返回一个JSON格式"})
