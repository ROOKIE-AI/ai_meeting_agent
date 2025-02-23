"""
工具类模块
包含各种工具类和辅助函数
"""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

class WikipediaToolWrapper:
    """维基百科搜索工具包装类"""
    
    def __init__(self):
        self.name = "wikipedia"
        self.description = "使用维基百科搜索信息的工具"
        self.wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh"))
    
    def func(self, query: str) -> str:
        try:
            return self.wiki_tool.run(query)
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"

def validate_api_key(api_key: str, api_base: str) -> bool:
    """
    验证API密钥是否有效
    
    Args:
        api_key: OpenAI API密钥
        api_base: API基础地址
        
    Returns:
        bool: 密钥是否有效
    """
    if not api_key or not api_key.startswith('sk-'):
        return False
    return True 