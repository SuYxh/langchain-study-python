# LangChain v0.3 模板项目

这是一个基于 LangChain v0.3 的模板项目，提供了统一的大语言模型（LLM）配置和初始化功能。该项目可以作为开发 LangChain 应用的起点，支持多种模型提供商，目前主要集成了 SiliconFlow 平台。

## 功能特点

- 统一的模型创建接口，支持多种模型提供商
- 内置 SiliconFlow 平台集成，支持 Qwen/Qwen3-8B 等模型
- 环境变量管理，通过 .env 文件配置 API 密钥
- 灵活的模型参数配置
- SSL 证书问题处理

## 项目结构

```
langchain-v0.3-template/
├── .env                 # 环境变量配置文件（需创建）
├── .env-example         # 环境变量示例文件
├── main.py              # 主要代码文件，包含模型初始化功能
├── pyproject.toml       # 项目配置和依赖管理
└── uv.lock              # uv 包管理器锁文件
```

## 技术栈

- **Python 3.12+**: 项目基础语言
- **LangChain v0.3**: 大语言模型应用开发框架
- **uv**: 高性能 Python 包管理器
- **python-dotenv**: 环境变量管理
- **httpx**: HTTP 客户端

## 安装说明

### 1. 克隆项目

```bash
git clone <项目仓库URL>
cd langchain-v0.3-template
```

### 2. 安装依赖

本项目使用 uv 作为包管理器，确保已安装 uv：

```bash
# 安装 uv（如果尚未安装）
pip install uv

# 安装项目依赖
uv sync
```

或者使用 pip：

```bash
pip install -e .
```

### 3. 配置环境变量

复制环境变量示例文件并配置必要的 API 密钥：

```bash
cp .env-example .env
# 编辑 .env 文件，添加你的 API 密钥
```

## 使用方法

### 基础使用

```python
from main import create_model

# 创建默认的 SiliconFlow 模型（Qwen/Qwen3-8B）
model = create_model()

# 调用模型生成回复
response = model.invoke("你好，请简单介绍一下自己")
print(response.content)
```

### 指定模型

```python
# 使用 SiliconFlow 平台的其他模型
model = create_model(provider="siliconflow", model_name="Qwen/Qwen3-72B")
```

### 直接使用 SiliconFlow 模型创建函数

```python
from main import create_siliconflow_model

model = create_siliconflow_model(model_name="Qwen/Qwen3-72B")
response = model.invoke("请解释量子计算的基本原理")
print(response.content)
```

## 配置指南

### 环境变量配置

在 `.env` 文件中配置以下环境变量：

```
# SiliconFlow API 密钥
SILICONFLOW_API_KEY=your_api_key_here
```

### 模型配置选项

目前支持的模型提供商：
- `siliconflow`: 硅基流动平台

### 默认模型

- SiliconFlow 平台默认使用 `Qwen/Qwen3-8B`

## 扩展项目

要添加新的模型提供商，请在 `main.py` 中：

1. 创建新的模型创建函数（参考 `create_siliconflow_model`）
2. 在 `create_model` 函数中添加对新提供商的支持

## 注意事项

1. 确保正确配置 API 密钥，否则模型初始化会失败
2. 本项目使用了 SSL 证书验证跳过的方式解决可能的证书问题，在生产环境中建议正确配置 SSL
3. Python 版本需要 3.12 或更高

## 依赖说明

本项目使用了丰富的 LangChain 生态系统组件，包括：

- langchain v0.3+
- langchain-community
- langchain-experimental
- langchain-openai
- langchain-ollama
- langgraph
- 以及其他辅助库

详细依赖列表请查看 `pyproject.toml` 文件。

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交 Issue 和 Pull Request。

## 联系方式

如有问题，请通过 GitHub Issues 与我们联系。