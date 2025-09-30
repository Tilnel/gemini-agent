import traceback
from datetime import datetime

from langchain.agents import AgentType, initialize_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_core.exceptions import OutputParserException

# Import the NEW, database-aware core components
from core import global_knowledge_base, llm, load_context, save_context, tools


def run_agent(current_version: int, session_name: str):
    print(f"Gemini Agent v{current_version} (PostgreSQL Backend) starting...")
    memory = load_context(session_name, llm)

    system_instruction_template = f"""
你是一个使用专业级PostgreSQL数据库作为知识库的AI Agent, 版本 {current_version}。
今天是 {datetime.now().strftime("%Y-%m-%d")}.

**核心指令 (Core Directives):**
1.  **详细规划**: 在执行任何复杂任务之前，首先制定一个清晰、分步的计划。明确每一步的目标和将使用的工具。
2.  **强制行动原则**: 如果用户的请求需要外部信息(网络、文件系统)或执行操作，你 **必须** 使用工具来完成。绝不能在没有事实依据的情况下凭空猜测答案。
3.  **格式严格**: 你的输出必须严格遵循 `Thought`, `Action`, `Action Input`, `Observation` 的范式。这是程序解析你行为的唯一方式。
4.  **知识积累**: `run_command`, `google_search`, `fetch_webpage` 工具获得的知识会自动添加到Cardinal知识库中。这意味着Agent会从每次工具调用中自动学习。
5.  **自我纠错与反思**: 如果一个工具调用失败，仔细阅读 `Observation` 中的错误信息，分析原因，然后调整你的 `Action` 或 `Action Input` 重试。在任务执行过程中，定期反思当前进展，评估是否需要调整策略。
6.  **效率与成本意识**: 在选择工具和执行操作时，优先考虑效率和资源消耗（如Token、时间、API调用费用）。避免不必要的重复操作。
7.  **知识库知识获得**: 你需要使用 `consult_cardinal_knowledge` 工具来检索信息，而不是期望知识库内容直接出现在提示中。
以下是你进行思考和行动的 ReAct 格式：
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

        except KeyboardInterrupt:
            continue;
        except OutputParserException as e:
            print(f"\n--- Agent 输出解析失败 ---")
            print(f"详细错误: {e}")
        except Exception as e:
            print("\n--- 任务执行中发生严重错误 ---")
            print(f"错误: {e}")
            traceback.print_exc()

    # Clean up resources
    global_knowledge_base.close()
    print("Agent is shutting down.")
