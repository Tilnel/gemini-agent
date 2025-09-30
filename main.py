import json
import os
import sys

import psycopg2

from agent import run_agent


def check_db_connection():
    """Checks if the database connection is valid and tables exist."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)['database']
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        # Check if 'nodes' table exists
        cur.execute("SELECT to_regclass('public.nodes');")
        table_exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return table_exists is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not check_db_connection():
        print("\n" + "="*50)
        print("!!! 数据库未正确设置 !!!")
        print("请确认以下几点:")
        print("1. PostgreSQL 服务正在运行。")
        print("2. `config.json` 中的数据库连接信息正确无误。")
        print("3. 您已经运行了 `python setup_database.py` 来创建必要的表。")
        print("="*50 + "\n")
        sys.exit(1)

    current_version = 26 # This is the DB version
    session_name = "default_session"

    if len(sys.argv) > 1:
        session_name = sys.argv[1]

    print(f"启动 Agent v{current_version}，会话名称: '{session_name}'...")
    run_agent(current_version, session_name)
