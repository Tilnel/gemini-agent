import json
import sys
import psycopg2


def check_db_connection():
    """Checks if the database connection is valid and tables exist (specifically 'nodes')."""
    conn = None
    try:
        with open('config.json', 'r') as f:
            # 1. 加载数据库配置
            config = json.load(f)['database']

        # 2. 尝试建立连接
        conn = psycopg2.connect(**config)
        cur = conn.cursor()

        # 3. 检查 'nodes' 表是否存在。to_regclass 在表不存在时返回 NULL。
        # 结果是一个包含 None 的元组 (None,)
        cur.execute("SELECT to_regclass('public.nodes');")
        table_exists_result = cur.fetchone()[0]

        # 4. 关键修正: 如果 table_exists_result 是 None，则表不存在。
        table_exists = table_exists_result is not None

        cur.close()
        conn.close()
        return table_exists

    except FileNotFoundError:
        # 如果 config.json 不存在
        return False
    except psycopg2.OperationalError as e:
        # 如果数据库连接失败、配置错误、服务未启动等
        print(f"数据库连接或操作失败: {e}", file=sys.stderr)
        return False
    except Exception:
        # 捕获其他未知错误，例如 JSON 解析错误
        return False
    finally:
        if conn:
            # 确保连接被关闭
            try:
                conn.close()
            except:
                pass




if __name__ == "__main__":
    if not check_db_connection():
        print("\n" + "=" * 50)
        print("!!! 数据库未正确设置 !!!")
        print("请确认以下几点:")
        print("1. PostgreSQL 服务正在运行。")
        print("2. `config.json` 中的数据库连接信息正确无误。")
        print("3. 您已经运行了 `python setup_database.py` 来创建必要的表。")
        print("=" * 50 + "\n")
        sys.exit(1)


from agent import run_agent

if __name__ == "__main__":
    current_version = 26  # This is the agent version
    session_name = "default_session"

    if len(sys.argv) > 1:
        session_name = sys.argv[1]

    print(f"启动 Agent v{current_version}，会话名称: '{session_name}'...")
    run_agent(current_version, session_name)
