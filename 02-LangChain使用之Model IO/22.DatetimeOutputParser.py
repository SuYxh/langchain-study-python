import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ( # pyright: ignore[reportMissingImports]
    DatetimeOutputParser, 
)  
from datetime import datetime
from pydantic import SecretStr

# 加载环境变量
dotenv.load_dotenv()

# 初始化模型
base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None

chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)

# 1. 创建 DatetimeOutputParser 实例
parser = DatetimeOutputParser()

# 2. 获取格式化指令
format_instructions = parser.get_format_instructions()
print("格式化指令:")
print(format_instructions)
print("\n" + "=" * 50 + "\n")

# 3. 创建提示词模板
prompt = PromptTemplate(
    template="回答用户的问题。\n{format_instructions}\n问题: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

# 4. 创建链
chain = prompt | chat_model | parser

# 5. 测试不同的问题
questions = [
    "中华人民共和国成立于什么时候？",
    # "第二次世界大战结束的具体日期是什么？",
]

for question in questions:
    try:
        print(f"问题: {question}")

        # 格式化提示词查看
        formatted_prompt = prompt.format(question=question)
        print("发送给模型的完整提示词:")
        print(formatted_prompt)
        print("\n" + "-" * 30 + "\n")

        # 调用链
        result = chain.invoke({"question": question})

        print(f"解析结果 (datetime对象): {result}")
        print(f"结果类型: {type(result)}")
        print(f"格式化显示: {result.strftime('%Y年%m月%d日 %H:%M:%S')}")

    except Exception as e:
        print(f"解析失败: {e}")

    print("\n" + "=" * 50 + "\n")
