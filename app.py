"""
会议准备AI助手应用主入口
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from app.main import main

if __name__ == "__main__":
    main()