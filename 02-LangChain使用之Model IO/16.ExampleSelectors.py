import os

# 设置环境变量来避免OpenMP重复初始化错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from pydantic import SecretStr


load_dotenv()


base_url = os.getenv("QINIU_BASE_URL", "")
model_name = os.getenv("QINIU_MODEL_GPT_OSS_20B", "")
api_key_str = os.getenv("QINIU_API_KEY", "")
api_key = SecretStr(api_key_str) if api_key_str else None


siliconflow_base_url = os.getenv("SILICONFLOW_BASE_URL", "")
siliconflow_embedding_model_name = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "")
siliconflow_api_key_str = os.getenv("SILICONFLOW_API_KEY", "")
siliconflow_api_key = (
    SecretStr(siliconflow_api_key_str) if siliconflow_api_key_str else None
)

# 初始化模型和嵌入模型
chat_model = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
embeddings = OpenAIEmbeddings(
    model=siliconflow_embedding_model_name,  # 指定模型名称
    api_key=siliconflow_api_key,  # 你的 API Key
    base_url=siliconflow_base_url,  # 硅基流动的 API 基础地址
)

# 1. 创建一个大的示例池（我们的“图书馆”）
examples = [
    {"input": "如何创建一个空列表？", "output": "my_list = []"},
    {"input": "如何向列表添加元素？", "output": "my_list.append(item)"},
    {
        "input": "定义一个计算圆面积的函数",
        "output": "def calculate_circle_area(radius):\n    return 3.14 * radius * radius",
    },
    {
        "input": "如何遍历一个字典？",
        "output": "for key, value in my_dict.items():\n    print(key, value)",
    },
    {
        "input": "创建一个包含三个元素的字典",
        "output": "my_dict = {'a': 1, 'b': 2, 'c': 3}",
    },
    {"input": "如何从列表中移除一个元素？", "output": "my_list.remove(item)"},
    {
        "input": "定义一个带默认参数的函数",
        "output": "def greet(name, greeting='Hello'):\n    print(f'{{greeting}}, {{name}}!')",
    },
    {
        "input": "如何检查一个键是否在字典中？",
        "output": "if 'key' in my_dict:\n    print('Key exists')",
    },
    {
        "input": "创建一个类来表示一个学生",
        "output": "class Student:\n    def __init__(self, name, id):\n        self.name = name\n        self.id = id",
    },
    {"input": "如何对列表进行排序？", "output": "my_list.sort()"},
]

# 2. 创建 Example Selector（我们的“智能管理员”）
# 它会使用 FAISS 来进行高效的语义搜索
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,  # 示例池
    embeddings,  # 用于将文本转换为向量的模型
    FAISS,  # 向量数据库，用于快速相似性搜索
    k=2,  # 为每个输入选择最相似的 2 个示例
)

# 3. 创建 FewShotPromptTemplate，但这次 examples 参数使用 selector
prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # 动态选择示例
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="用户需求: {input}\n代码示例: {output}",
    ),
    prefix="你是一个专业的Python程序员。请根据用户的需求，参考提供的代码示例，生成相应的Python代码。",
    suffix="用户需求: {input}\n代码示例:",
    input_variables=["input"],
)

# 4. 创建链
chain = prompt_template | chat_model


# 5. 测试不同的用户输入
def test_and_explain(user_input):
    print(f"\n{'='*50}")
    print(f"用户输入: {user_input}")

    # 查看被选中的示例
    selected_examples = example_selector.select_examples({"input": user_input})
    print("\n--- 为该输入选择的示例 ---")
    for i, ex in enumerate(selected_examples):
        print(f"示例 {i+1}:")
        print(f"  需求: {ex['input']}")
        print(f"  代码: {ex['output']}")

    # 调用链并获取最终响应
    response = chain.invoke({"input": user_input})
    print("\n--- AI 生成的代码 ---")
    print(response.content)


# 测试案例1：与列表相关
test_and_explain("我该如何清空一个列表？")

# 测试案例2：与函数定义相关（语义匹配）
# test_and_explain("写个函数算矩形周长")

# 测试案例3：与字典相关
# test_and_explain("怎么获取字典里的值？")
