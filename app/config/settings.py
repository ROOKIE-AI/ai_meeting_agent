"""
应用配置文件
包含所有全局配置和常量
"""

import os

# OpenAI模型配置
MODEL_OPTIONS = {
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-o1-mini": "o1-mini",
    "GPT-o1": "o1-preview",
    "GPT-o3-mini": "o3-mini",
}

# 默认API设置
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_TEMPERATURE = 0.7

# 文档处理配置
SUPPORTED_EXTENSIONS = {
    'text': ['.txt', '.md'],
    'document': ['.doc', '.docx'],
    'spreadsheet': ['.xls', '.xlsx', '.csv'],
    'presentation': ['.ppt', '.pptx'],
    'pdf': ['.pdf'],
    'image': ['.jpg', '.jpeg', '.png'],
    'archive': ['.zip', '.rar']
}

# 会议设置
MIN_MEETING_DURATION = 15
MAX_MEETING_DURATION = 180
DEFAULT_MEETING_DURATION = 60
MEETING_DURATION_STEP = 15

# 系统提示词
SYSTEM_PROMPT = "你是一位专业的AI助手。请始终使用中文回答所有问题。确保输出的所有内容都是中文，包括标题、分析和建议。" 