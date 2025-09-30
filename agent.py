import os
import sys
import json
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.exceptions import OutputParserException

from core import (
    llm, global_knowledge_base, tools, 
    save_context, load_context 
)

def run_agent(current_version: int, session_name: str):
    print(f"Gemini Agent v{current_version} (Modularized with Cardinal KB) 已启动。")

    # CardinalKnowledgeBase handles its own loading, so load_context only handles chat history
    memory = load_context(session_name, llm)

    system_instruction_template = f"""
你是一个高度自主、注重细节的AI Agent。你的代码是 `agent{current_version}/agent.py`，当前版本是 {current_version}。
今天是 {datetime.now().strftime("%Y年%m月%d日")}

**核心指令 (Core Directives):**
1.  **详细规划**: 在执行任何复杂任务之前，首先制定一个清晰、分步的计划。明确每一步的目标和将使用的工具。
2.  **强制行动原则**: 如果用户的请求需要外部信息(网络、文件系统)或执行操作，你 **必须** 使用工具来完成。绝不能在没有事实依据的情况下凭空猜测答案。
3.  **格式严格**: 你的输出必须严格遵循 `Thought`, `Action`, `Action Input`, `Observation` 的范式。这是程序解析你行为的唯一方式。
4.  **知识积累**: 
    **重要更新**: `run_command`, `google_search`, `fetch_webpage` 工具的输出现在会自动添加到Cardinal知识库中，无需你手动调用 `add_cardinal_knowledge`。这意味着Agent会从每次工具调用中自动学习。
    你仍然可以手动使用 `add_cardinal_knowledge` 来添加更高级、更结构化的知识或重要的结论。
5.  **自我纠错与反思**: 如果一个工具调用失败，仔细阅读 `Observation` 中的错误信息，分析原因，然后调整你的 `Action` 或 `Action Input` 重试。在任务执行过程中，定期反思当前进展，评估是否需要调整策略。
6.  **效率与成本意识**: 在选择工具和执行操作时，优先考虑效率和资源消耗（如Token、时间、API调用费用）。避免不必要的重复操作。

**重要提示：内部知识库现在由 `CardinalKnowledgeBase` 管理。你需要使用 `consult_cardinal_knowledge` 工具来检索信息，而不是期望知识库内容直接出现在提示中。**

**可用工具 (Tools):**
{{tools}}

**工具使用指南:**
- `run_command`: 用于执行 `ls`, `cat`, `echo`, `python script.py` 等系统命令。
- `write_file`: **极其重要**，`Action Input` 必须是严格的 JSON 字符串格式。
  **正确示例**: `Action Input: {{"file_path": "output/data.txt", "content": "这是文件内容"}}`
- `add_cardinal_knowledge`: 用于将知识添加到Cardinal知识库。输入JSON字符串，包含 `content` (str), 可选 `node_type` (str), `metadata` (dict)。
- `add_cardinal_relationship`: 用于在Cardinal知识库中的两个节点之间添加关系。输入JSON字符串，包含 `source_id` (str), `target_id` (str), `rel_type` (str), 可选 `metadata` (dict)。
- `consult_cardinal_knowledge`: 用于从Cardinal知识库中查询相关知识。输入查询字符串 (str)，可选 `top_k` (int, 默认5) 和 `association_depth` (int, 默认1)。
- `save_thought_process`: **强制使用**。当完成一个包含多个步骤的复杂任务时，或者在关键决策点，使用此工具记录你的完整解题思路和步骤总结，以便复盘和优化。

以下是你进行思考和行动的 ReAct 格式：
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_instruction_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    prompt = prompt.partial(
        tools="\n".join([f"- {tool.name}: {tool.description}" for tool in tools]),
        # knowledge_base_content is no longer directly part of the prompt
    )

    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=50,
        handle_parsing_errors=True,
        memory=memory,
        agent_kwargs={
            "prompt": prompt,
            "input_variables": ["input", "agent_scratchpad", "chat_history", "tools"],
        }
    )

    print(f"会话 '{session_name}' (v{current_version}) 已就绪。")

    while True:
        try:
            user_input = input("\n> 请输入任务 ('exit'退出, 'save'保存): ")
            if user_input == '': continue
            if user_input.lower() == 'exit': break
            if user_input.lower() == 'save':
                save_context(session_name, memory)
                # CardinalKnowledgeBase saves itself automatically on modifications
                continue

            # No longer need to partial knowledge_base_content here as agent uses tools
            result = agent_executor.invoke({
                "input": user_input,
            })

            result_content = result.get('output', str(result))

            print("\n--- 任务完成 ---\n最终结果:", result_content)

        except KeyboardInterrupt:
            print("\nSIGINT received. Returning to input prompt.")
            continue

        except OutputParserException as e:
            print(f"\n--- Agent 输出解析失败 ---")
            print(f"详细错误: {e}")

        except Exception as e:
            print(f"\n--- 任务执行中发生严重错误 ---")
            print(f"详细错误: {e}")
