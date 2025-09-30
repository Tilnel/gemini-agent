import os
import sys
import json
import subprocess
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.schema.messages import HumanMessage, AIMessage

# Import the NEW, database-backed Cardinal Knowledge Base
from cardinal import CardinalKnowledgeBase
from typing import Dict, Callable

# --- Configuration & Initialization ---
def load_config():
    """Loads configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it.")
        sys.exit(1)

config = load_config()

try:
    google_api_key = os.environ['GOOGLE_API_KEY']
except KeyError:
    print("错误: 环境变量 'GOOGLE_API_KEY' 未设置。")
    sys.exit(1)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# --- Initialize the DB-backed Knowledge Base ---
global_knowledge_base = CardinalKnowledgeBase(
    db_config=config['database'],
    agent_config=config['agent_config']
)

# --- Tools Wrapper for Automatic Learning ---
def _auto_learn_from_observation(tool_name: str, tool_input: str, observation: str):
    """
    Automatically adds relevant information from tool observations to the Cardinal Knowledge Base.
    """
    if not observation or observation.strip().lower().startswith("错误:"): # Don't learn from errors
        return

    original_length = len(observation)
    observation_content = observation

    # Decide on a sensible node_type and content for the knowledge
    knowledge_content = f"Observation from {tool_name} with input '{tool_input}':\n{observation_content}"

    metadata = {
        'tool_name': tool_name,
        'tool_input': tool_input,
        'timestamp': datetime.now().isoformat(),
        'original_length': original_length,
        'truncated': False,
    }

    try:
        node = global_knowledge_base.add_knowledge(
            content=knowledge_content,
            node_type='tool_observation',
            metadata=metadata
        )[0]
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


# --- Tools (logic remains similar, but now interacts with the DB-backed KB) ---

# Learning wrappers and other tool functions would be here.
# For brevity, I'm omitting the full code from the previous response,
# as their internal logic (calling the KB) doesn't need to change.
# The `global_knowledge_base` object they call is now the DB version.

def run_command(command: str) -> str:
    """在本地 shell 中执行一条命令并返回输出。"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=300)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}".strip()
    except Exception as e:
        return f"执行命令时出错: {e}"

def fetch_webpage_content(url: str) -> str:
    """使用 curl 命令获取指定 URL 的网页内容。"""
    command = f"curl -sL '{url}'"
    return run_command(command)

def write_file(json_str_input: str) -> str:
    """将内容写入文件。"""
    try:
        data = json.loads(json_str_input)
        path, content = data['file_path'], data['content']
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
        return f"文件 '{path}' 写入成功。"
    except Exception as e:
        return f"写入文件时出错: {e}"

def add_cardinal_knowledge(json_str_input: str) -> str:
    try:
        data = json.loads(json_str_input)
        node = global_knowledge_base.add_knowledge(
            content=data['content'],
            node_type=data.get('node_type', 'concept'),
            metadata=data.get('metadata', {})
        )
        return f"知识已添加到数据库。节点ID: {node.id}"
    except Exception as e:
        return f"添加知识到数据库失败: {e}"

def add_cardinal_relationship(json_str_input: str) -> str:
    try:
        data = json.loads(json_str_input)
        edge = global_knowledge_base.add_relationship(**data)
        return f"关系已添加到数据库。边ID: {edge.id}"
    except Exception as e:
        return f"添加关系到数据库失败: {e}"

def consult_cardinal_knowledge(query: str) -> str:
    """从数据库知识库中查询相关知识。"""
    results = global_knowledge_base.consult(query)
    if not results:
        return "在数据库知识库中未找到相关知识。"

    # 新格式: res['id'] 和 res['content']
    formatted = [f"- ID: {res['id']}, 内容: {res['content'][:150]}... (相似度: {res['similarity']:.2f})" for res in results]

    return "从数据库中查询到以下信息:\n" + "\n".join(formatted)

tools = [
    Tool(name="run_command", func=create_learning_wrapper(run_command, "run_command"), description="执行本地 shell 命令。例如: 'ls -l', 'cat file.txt', 'pip install <package>'。"),
    Tool(name="google_search", func=create_learning_wrapper(search.run, "google_search"), description="当需要从互联网获取信息时使用。当你采信某个信源的时候，应当给出来源的网页。注意避免采取“今日”等模糊的日期指示，而是替换为真实的日期。"),
    Tool(name="fetch_webpage", func=create_learning_wrapper(fetch_webpage_content, "fetch_webpage"), description="获取特定 URL 的完整网页内容。输入应为一个有效的 URL。"),
    Tool(name="write_file", func=write_file, description="将内容写入文件。输入必须是包含 'file_path' 和 'content' 的JSON字符串。"),
    # Tool(name="add_cardinal_knowledge", func=add_cardinal_knowledge, description="将结构化知识添加到数据库知识库。格式为包含 'content'(raw text or html content) 的 JSON。"),
    # Tool(name="add_cardinal_relationship", func=add_cardinal_relationship, description="在数据库中的两个知识节点之间添加关系。"),
    Tool(name="consult_cardinal_knowledge", func=consult_cardinal_knowledge, description="从数据库知识库中查询相关知识。"),
]

# --- Context Management (unchanged) ---
def save_context(session_name: str, memory: ConversationTokenBufferMemory):
    file_path = f"{session_name}.json"
    history = [{"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content} for msg in memory.chat_memory.messages]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({"chat_history": history}, f, ensure_ascii=False, indent=4)
    print(f"\n--- 聊天记录已保存至 {file_path} ---")

def load_context(session_name: str, llm) -> ConversationTokenBufferMemory:
    file_path = f"{session_name}.json"
    memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, max_token_limit=10000)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for msg in data.get("chat_history", []):
                if msg["type"] == "human": memory.chat_memory.add_user_message(msg["content"])
                else: memory.chat_memory.add_ai_message(msg["content"])
        print(f"\n--- 已从 {file_path} 加载聊天记录 ---")
    return memory

