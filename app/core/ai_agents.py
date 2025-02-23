"""
AI助手模块
包含AI助手和团队的核心类
"""

from crewai import Agent, Crew
from typing import List
import streamlit as st

class LoggingAgent(Agent):
    """
    带日志记录功能的AI助手基类。
    继承自CrewAI的Agent类，添加了任务执行过程的日志记录功能。
    """
    
    _thoughts_store = {}  # 使用类变量存储所有实例的思考过程
    _analysis_store = {}  # 使用类变量存储所有实例的分析内容
    
    def __init__(self, *args, **kwargs):
        """初始化LoggingAgent实例"""
        super().__init__(*args, **kwargs)
        self._thoughts_store[self.role] = []
        self._analysis_store[self.role] = []
    
    @property
    def thoughts(self):
        """获取当前实例的思考过程"""
        return self._thoughts_store.get(self.role, [])
    
    @property
    def analysis(self):
        """获取当前实例的分析内容"""
        return self._analysis_store.get(self.role, [])
    
    def add_thought(self, thought: str):
        """添加一条思考记录"""
        if self.role not in self._thoughts_store:
            self._thoughts_store[self.role] = []
        self._thoughts_store[self.role].append(thought)
    
    def add_analysis(self, analysis: str):
        """添加一条分析记录"""
        if self.role not in self._analysis_store:
            self._analysis_store[self.role] = []
        self._analysis_store[self.role].append(analysis)
        
    def execute_task(self, task, context=None, **kwargs):
        """执行任务并记录执行过程"""
        try:
            task_description = task.description.split('\n')[1].strip()
            thought = f"🤖 **{self.role}** 开始新任务：\n> {task_description}"
            self.add_thought(thought)
            st.session_state.analysis_log.append(thought)
            
            thought = "🔍 正在搜索相关信息..."
            self.add_thought(thought)
            st.session_state.analysis_log.append(thought)
            
            result = super().execute_task(task, context=context, **kwargs)
            
            # 记录分析内容
            self.add_analysis(result)
            
            thought = f"✅ **{self.role}** 已完成分析"
            self.add_thought(thought)
            st.session_state.analysis_log.append(thought)
            
            # 保存分析结果
            if '背景分析' in self.role:
                st.session_state.expert_analysis['context'] = {
                    'result': result, 
                    'thoughts': self.thoughts,
                    'analysis': self.analysis
                }
            elif '行业专家' in self.role:
                st.session_state.expert_analysis['industry'] = {
                    'result': result, 
                    'thoughts': self.thoughts,
                    'analysis': self.analysis
                }
            elif '策略专家' in self.role:
                st.session_state.expert_analysis['strategy'] = {
                    'result': result, 
                    'thoughts': self.thoughts,
                    'analysis': self.analysis
                }
            elif '沟通专家' in self.role:
                st.session_state.expert_analysis['briefing'] = {
                    'result': result, 
                    'thoughts': self.thoughts,
                    'analysis': self.analysis
                }
            
            return result
        except Exception as e:
            thought = f"❌ **{self.role}** 执行任务时出错：{str(e)}"
            self.add_thought(thought)
            st.session_state.analysis_log.append(thought)
            raise e

    def __del__(self):
        """清理实例时移除记录"""
        if self.role in self._thoughts_store:
            del self._thoughts_store[self.role]
        if self.role in self._analysis_store:
            del self._analysis_store[self.role]

class LoggingCrew(Crew):
    """
    带日志记录功能的AI团队类。
    继承自CrewAI的Crew类，添加了团队工作过程的日志记录功能。
    """
    
    def __init__(self, *args, **kwargs):
        """初始化LoggingCrew实例"""
        super().__init__(*args, **kwargs)
        
    def kickoff(self):
        """启动团队工作并记录工作过程"""
        st.session_state.analysis_log.append("🚀 **AI分析团队启动**\n")
        st.session_state.analysis_log.append("📋 **工作计划**：")
        st.session_state.analysis_log.append("1. 分析会议背景和公司信息")
        st.session_state.analysis_log.append("2. 研究行业趋势和市场机会")
        st.session_state.analysis_log.append("3. 制定会议策略和议程")
        st.session_state.analysis_log.append("4. 生成执行简报\n")
        return super().kickoff() 