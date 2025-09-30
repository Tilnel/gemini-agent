import sys
import os

from agent import run_agent

if __name__ == "__main__":
    current_version = 25 # This is agent25
    session_name = "default_session_25"

    if len(sys.argv) > 1:
        session_name = sys.argv[1]

    print(f"启动 Agent v{current_version}，会话名称: '{session_name}'...")
    run_agent(current_version, session_name)
