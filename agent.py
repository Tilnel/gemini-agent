from datetime import datetime

from langchain.agents import AgentType, initialize_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage

# Import the NEW, database-aware core components
from core import global_knowledge_base, llm, load_context, save_context, tools


def run_agent(current_version: int, session_name: str):
    print(f"Gemini Agent v{current_version} (PostgreSQL Backend) starting...")
    memory = load_context(session_name, llm)

    system_instruction_template = f"""
你是一个使用专业级PostgreSQL数据库作为知识库的AI Agent, 版本 {current_version}。
今天是 {datetime.now().strftime("%Y-%m-%d")}.
你的核心指令和工具使用方法保持不变，但请知悉你的所有知识都将被持久化到一个稳定、可扩展的数据库中。
"""
    # System prompt can be reused from the previous version

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    prompt = prompt.partial(tools="\n".join([
        f"- {tool.name}: {tool.description}" for tool in tools
    ]))

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=25,
        handle_parsing_errors=True,
        memory=memory,
        agent_kwargs={"prompt": prompt}
    )

    print(f"会话 '{session_name}' (v{current_version}) 已就绪。")
    while True:
        try:
            user_input = input("\n> 请输入任务/问题 ('exit'退出): ")
            if user_input.lower() == 'exit':
                break
            if not user_input:
                continue

            result = agent_executor.invoke({"input": user_input})
            print("\n--- 任务完成 ---")
            print("最终结果:", result.get('output', str(result)))
            save_context(session_name, memory)
            # Save chat history after each turn

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print("\n--- 任务执行中发生严重错误 ---")
            print(f"错误: {e}")

    # Clean up resources
    global_knowledge_base.close()
    print("Agent is shutting down.")
