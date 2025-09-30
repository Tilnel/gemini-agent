import os
import sys
import json
import subprocess
import re
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema.agent import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
import ast
from typing import Dict, Any, List, Optional, Callable # Added Callable

# Import Cardinal Knowledge Base components
from cardinal_api import CardinalKnowledgeBase, Node, Edge # Changed from agent24 to agent25

# --- API Key 和 LLM 配置 ---
try:
    google_api_key = os.environ['GOOGLE_API_KEY']
except KeyError:
    print("错误: 环境变量 'GOOGLE_API_KEY' 未设置。")
    sys.exit(1)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_output_tokens=50000)

# --- 内部知识库 (现在使用 Cardinal) ---
global_knowledge_base = CardinalKnowledgeBase()

# --- 工具定义 (Tools) ---
def run_command(command: str) -> str:
    """在本地 shell 中执行一条命令并返回输出。"""
    try:
        result = subprocess.run(
            command, shell=True, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=300
        )
        output = f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output
    except subprocess.CalledProcessError as e:
        return f"命令执行失败。返回码: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    except Exception as e:
        return f"执行时出现未知错误: {e}"

def fetch_webpage_content(url: str) -> str:
    """使用 curl 命令获取指定 URL 的网页内容。"""
    command = f"curl -sL '{url}'"
    return run_command(command)

def write_file(json_str_input: str) -> str:
    """将指定内容写入文件。输入必须是包含 'file_path' 和 'content' 的JSON字符串。"""
    try:
        data = json.loads(json_str_input)
        file_path = data.get('file_path')
        content = data.get('content')
        if not isinstance(file_path, str) or not file_path:
            return "错误: JSON 输入中缺少 'file_path' 或其值无效。"
        if not isinstance(content, str):
            return "错误: JSON 输入中缺少 'content' 或其值不是字符串。"

        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件 '{file_path}' 写入成功。"
    except json.JSONDecodeError:
        return f"错误：Action Input 不是有效的 JSON 字符串: {json_str_input}"
    except Exception as e:
        return f"写入文件时发生未知错误: {e}"

def add_cardinal_knowledge(json_str_input: str) -> str:
    """
    将知识添加到Cardinal知识库。
    输入必须是包含 'content' 的JSON字符串，可选 'node_type' 和 'metadata'。
    示例: {{\"content\": \"Python是一种高级编程语言\", \"node_type\": \"编程语言\", \"metadata\": {{\"version\": \"3.9\"}}}}}}
    """
    try:
        data = json.loads(json_str_input)
        content = data.get('content')
        node_type = data.get('node_type', 'concept')
        metadata = data.get('metadata', {})

        if not isinstance(content, str) or not content:
            return "错误: JSON 输入中缺少 'content' 或其值无效。"

        node = global_knowledge_base.add_knowledge(content=content, node_type=node_type, metadata=metadata)
        return f"知识已添加到Cardinal知识库。节点ID: {node.id}, 内容: {node.content[:50]}..."
    except json.JSONDecodeError:
        return f"错误：Action Input 不是有效的 JSON 字符串: {json_str_input}"
    except Exception as e:
        return f"添加Cardinal知识失败: {e}"

def add_cardinal_relationship(json_str_input: str) -> str:
    """
    在Cardinal知识库中的两个节点之间添加关系。
    输入必须是包含 'source_id', 'target_id', 'rel_type' 的JSON字符串，可选 'metadata'。
    示例: {{\"source_id\": \"node-123\", \"target_id\": \"node-456\", \"rel_type\": \"has_feature\", \"metadata\": {{\"strength\": 0.8}}}}}}
    """
    try:
        data = json.loads(json_str_input)
        source_id = data.get('source_id')
        target_id = data.get('target_id')
        rel_type = data.get('rel_type')
        metadata = data.get('metadata', {})

        if not all([source_id, target_id, rel_type]):
            return "错误: JSON 输入中缺少 'source_id', 'target_id' 或 'rel_type'。"

        edge = global_knowledge_base.add_relationship(source_id=source_id, target_id=target_id, rel_type=rel_type, metadata=metadata)
        if edge:
            return f"关系已添加到Cardinal知识库。边ID: {edge.id}, 类型: {edge.type}, 从 {edge.source_node_id} 到 {edge.target_node_id}"
        else:
            return f"添加Cardinal关系失败，可能因为节点ID不存在。"
    except json.JSONDecodeError:
        return f"错误：Action Input 不是有效的 JSON 字符串: {json_str_input}"
    except Exception as e:
        return f"添加Cardinal关系失败: {e}"

def consult_cardinal_knowledge(query: str, top_k: int = 5, association_depth: int = 1) -> str:
    """
    从Cardinal知识库中查询相关知识。
    输入为查询字符串，可选 'top_k' (返回数量) 和 'association_depth' (关联深度)。
    示例: \"关于Python编程语言的知识\"
    """
    try:
        results = global_knowledge_base.consult(query=query, top_k=top_k, association_depth=association_depth)
        if not results:
            return "在Cardinal知识库中未找到相关知识。"
        
        formatted_results = []
        for res in results:
            node = res['node']
            similarity_info = f" (相似度: {res['similarity']:.2f})" if 'similarity' in res else ""
            formatted_results.append(
                f"- ID: {node.id}\n  类型: {node.type}\n  内容: {node.content}\n  元数据: {node.metadata}{similarity_info}"
            )
        return "从Cardinal知识库中查询到以下信息:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"查询Cardinal知识失败: {e}"

def save_thought_process(reasoning_summary: str) -> str:
    """
    当完成一个复杂的、多步骤的任务后，用于将整个思考和行动链条的摘要保存到单独的日志文件中。
    """
    try:
        log_dir = "session_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"thought_process_{timestamp}.log")

        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(reasoning_summary)

        return f"详细的思考过程已保存至 {log_file}"
    except Exception as e:
        return f"保存思考过程失败: {e}"

# --- Tools Wrapper for Automatic Learning ---
def _auto_learn_from_observation(tool_name: str, tool_input: str, observation: str):
    """
    Automatically adds relevant information from tool observations to the Cardinal Knowledge Base.
    """
    if not observation or observation.strip().lower().startswith("错误:"): # Don't learn from errors
        return

    # Truncate long observations to avoid overwhelming the KB and embedding model
    max_content_length = 2000
    original_length = len(observation)
    truncated = False
    if original_length > max_content_length:
        observation_content = observation[:max_content_length] + "\n... (truncated)"
        truncated = True
    else:
        observation_content = observation

    # Decide on a sensible node_type and content for the knowledge
    knowledge_content = f"Observation from {tool_name} with input '{tool_input}':\n{observation_content}"
    
    metadata = {
        'tool_name': tool_name,
        'tool_input': tool_input,
        'timestamp': datetime.now().isoformat(),
        'original_length': original_length,
        'truncated': truncated,
    }

    try:
        node = global_knowledge_base.add_knowledge(
            content=knowledge_content, 
            node_type='tool_observation', 
            metadata=metadata
        )
        print(f"[Auto-Learning] Added knowledge from {tool_name} (ID: {node.id})")
    except Exception as e:
        print(f"[Auto-Learning Error] Failed to add knowledge from {tool_name}: {e}")

def create_learning_wrapper(original_func: Callable, tool_name: str) -> Callable:
    """
    Creates a wrapper around a tool's function to automatically learn from its observation.
    """
    def wrapper(*args, **kwargs):
        # Determine tool_input based on common patterns
        tool_input = ""
        if args:
            tool_input = str(args[0]) # Most tools take a single string argument
        elif tool_name == "run_command" and 'command' in kwargs:
            tool_input = kwargs['command']
        elif tool_name == "google_search" and 'query' in kwargs:
            tool_input = kwargs['query']
        elif tool_name == "fetch_webpage" and 'url' in kwargs:
            tool_input = kwargs['url']

        # Execute the original tool function
        observation = original_func(*args, **kwargs)

        # Trigger automatic learning
        _auto_learn_from_observation(tool_name, tool_input, observation)
        
        return observation
    return wrapper


search = GoogleSearchAPIWrapper()

tools = [
    Tool(name="run_command", func=create_learning_wrapper(run_command, "run_command"), description="执行本地 shell 命令。例如: 'ls -l', 'cat file.txt', 'pip install <package>'。"),
    Tool(name="google_search", func=create_learning_wrapper(search.run, "google_search"), description="当需要从互联网获取信息时使用。"),
    Tool(name="fetch_webpage", func=create_learning_wrapper(fetch_webpage_content, "fetch_webpage"), description="获取特定 URL 的完整网页内容。输入应为一个有效的 URL。"),
    Tool(name="write_file", func=write_file, description="将内容写入文件。输入必须是包含 'file_path' 和 'content' 的JSON字符串。"),
    Tool(name="add_cardinal_knowledge", func=add_cardinal_knowledge, description="将结构化知识添加到Cardinal知识库。输入JSON字符串，包含 'content' (str), 可选 'node_type' (str), 'metadata' (dict)。"),
    Tool(name="add_cardinal_relationship", func=add_cardinal_relationship, description="在Cardinal知识库中的两个节点之间添加关系。输入JSON字符串，包含 'source_id' (str), 'target_id' (str), 'rel_type' (str), 可选 'metadata' (dict)。"),
    Tool(name="consult_cardinal_knowledge", func=consult_cardinal_knowledge, description="从Cardinal知识库中查询相关知识。输入查询字符串 (str)，可选 'top_k' (int, 默认5) 和 'association_depth' (int, 默认1)。"),
    Tool(name="save_thought_process", func=save_thought_process, description="保存对一个复杂任务的完整解决思路和步骤总结，以便复盘。")
]

# --- 上下文管理 ---
def save_context(session_name: str, memory: ConversationTokenBufferMemory):
    """保存详细的对话历史。Cardinal知识库自动管理其持久化。"""
    file_path = f"{session_name}.json"

    chat_history_messages = memory.chat_memory.messages
    history_list = []
    for msg in chat_history_messages:
        if isinstance(msg, HumanMessage):
            history_list.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history_list.append({"type": "ai", "content": msg.content})

    data_to_save = {
        "chat_history": history_list,
        # Cardinal knowledge base manages its own persistence, no need to save it here.
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)
    print(f"\n--- 详细会话历史已保存至 {file_path} ---")

def load_context(session_name: str, llm) -> ConversationTokenBufferMemory:
    """从文件中加载详细的对话历史。Cardinal知识库在实例化时自动加载。"""
    file_path = f"{session_name}.json"
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, max_token_limit=6000)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            history_list = data.get("chat_history", [])
            for msg in history_list:
                if msg["type"] == "human":
                    memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    memory.chat_memory.add_ai_message(msg["content"])
        print(f"\n--- 已加载详细会话历史: {session_name} ---")
    else:
        print(f"\n--- 未找到会话文件 '{file_path}'，将开始新会话 ---")

    return memory
