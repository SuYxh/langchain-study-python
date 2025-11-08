from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ChatMessage, ToolMessage, FunctionMessage

# 创建不同类型的消息
system_msg = SystemMessage(content="你是一个专业的助手")
human_msg = HumanMessage(content="你好")
ai_msg = AIMessage(content="你好！有什么我可以帮助你的吗？")
chat_msg = ChatMessage(role="user", content="你好")
tool_msg = ToolMessage(content="你好！有什么我可以帮助你的吗？", tool_call_id="123")
function_msg = FunctionMessage(content="你好！有什么我可以帮助你的吗？", name="function_name")



# 打印消息类型
print(system_msg.content,system_msg.type)  # 输出: system
print(human_msg.content,human_msg.type)  # 输出: human
print(ai_msg.content,ai_msg.type)  # 输出: ai
print(chat_msg.content,chat_msg.type)  # 输出: user
print(tool_msg.content,tool_msg.type)  # 输出: tool
print(function_msg.content,function_msg.type)  # 输出: function
