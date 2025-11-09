from langchain_core.prompts import load_prompt

prompt = load_prompt("asset/prompt.json", encoding="utf-8")

print(prompt.format(name="张三", what="搞笑的"))
