"""
AI会议准备助手应用
这个模块实现了一个基于Streamlit的AI会议准备助手，利用OpenAI的GPT模型和维基百科搜索功能，
以及DuckDuckGo搜索，生成全面的会议准备材料。系统通过多个AI助手协同工作，提供会议背景分析、
行业趋势分析、会议策略制定和执行简报生成等多种功能。

主要功能：
    - 多AI助手协同工作系统
    - 实时分析进度展示
    - 中文维基百科集成
    - 可配置的AI模型参数

Author: Rookie
Date: 2025-02-20
"""

__import__('pysqlite3')  
import sys  
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.process import Process
# from langchain.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from typing import Any
import os

# 添加工具包装类
class WikipediaToolWrapper:
    def __init__(self):
        self.name = "wikipedia"
        self.description = "使用维基百科搜索信息的工具"
        self.wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh"))
    
    def func(self, query: str) -> str:
        try:
            return self.wiki_tool.run(query)
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"

class DuckDuckGoToolWrapper:
    def __init__(self):
        self.name = "duckduckgo"
        self.description = "使用DuckDuckGo搜索最新信息的工具"
        self.search_tool = DuckDuckGoSearchRun()
    
    def func(self, query: str) -> str:
        try:
            return self.search_tool.run(query)
        except Exception as e:
            return f"搜索时发生错误: {str(e)}"

# Streamlit 应用设置
st.set_page_config(page_title="AI会议助手 📝", layout="wide")
st.title("AI会议准备助手 📝")

# 创建分析过程显示区域
if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []
    
# 在session_state中添加导航状态
if 'current_view' not in st.session_state:
    st.session_state.current_view = "input"

# 添加导航栏
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
with nav_col1:
    if st.button("📝 输入信息", use_container_width=True):
        st.session_state.current_view = "input"
with nav_col2:
    if st.button("🔍 会议背景", use_container_width=True):
        st.session_state.current_view = "context"
with nav_col3:
    if st.button("📈 行业趋势", use_container_width=True):
        st.session_state.current_view = "industry"
with nav_col4:
    if st.button("📋 会议策略", use_container_width=True):
        st.session_state.current_view = "strategy"
with nav_col5:
    if st.button("📑 执行简报", use_container_width=True):
        st.session_state.current_view = "briefing"

st.divider()

# 在session_state中添加专家分析结果存储
if 'expert_analysis' not in st.session_state:
    st.session_state.expert_analysis = {
        'context': {'result': '', 'thoughts': [], 'analysis': []},
        'industry': {'result': '', 'thoughts': [], 'analysis': []},
        'strategy': {'result': '', 'thoughts': [], 'analysis': []},
        'briefing': {'result': '', 'thoughts': [], 'analysis': []}
    }

def add_log(message: str) -> None:
    """
    向分析日志中添加新的消息。

    Args:
        message (str): 要添加到日志的消息内容

    Returns:
        None
    """
    st.session_state.analysis_log.append(message)

def display_logs() -> None:
    """
    显示所有分析日志消息。
    使用Streamlit容器来展示日志内容，保持界面整洁。

    Returns:
        None
    """
    log_placeholder = st.empty()
    with log_placeholder.container():
        for log in st.session_state.analysis_log:
            st.markdown(log)

class LoggingAgent(Agent):
    """
    带日志记录功能的AI助手基类。
    继承自CrewAI的Agent类，添加了任务执行过程的日志记录功能。

    Attributes:
        role (str): AI助手的角色名称
        goal (str): AI助手的目标
        backstory (str): AI助手的背景故事
        verbose (bool): 是否输出详细日志
        allow_delegation (bool): 是否允许任务委派
        tools (list): AI助手可用的工具列表
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
        """
        执行任务并记录执行过程。

        Args:
            task: 要执行的任务对象
            context: 任务执行的上下文信息
            **kwargs: 其他参数

        Returns:
            str: 任务执行结果

        Raises:
            Exception: 任务执行过程中的错误
        """
        try:
            task_description = task.description.split('\n')[1].strip()
            thought = f"🤖 **{self.role}** 开始新任务：\n> {task_description}"
            self.add_thought(thought)
            add_log(thought)
            
            thought = "🔍 正在搜索相关信息..."
            self.add_thought(thought)
            add_log(thought)
            
            result = super().execute_task(task, context=context, **kwargs)
            
            # 记录分析内容
            self.add_analysis(result)
            
            thought = f"✅ **{self.role}** 已完成分析"
            self.add_thought(thought)
            add_log(thought)
            
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
            add_log(thought)
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

    Attributes:
        agents (list): AI助手团队成员列表
        tasks (list): 要执行的任务列表
        verbose (bool): 是否输出详细日志
        process (Process): 任务处理方式
    """
    
    def __init__(self, *args, **kwargs):
        """初始化LoggingCrew实例"""
        super().__init__(*args, **kwargs)
        
    def kickoff(self):
        """
        启动团队工作并记录工作过程。

        Returns:
            str: 团队工作的最终结果
        """
        add_log("🚀 **AI分析团队启动**\n")
        add_log("📋 **工作计划**：")
        add_log("1. 分析会议背景和公司信息")
        add_log("2. 研究行业趋势和市场机会")
        add_log("3. 制定会议策略和议程")
        add_log("4. 生成执行简报\n")
        return super().kickoff()

# 侧边栏API密钥设置
with st.sidebar:
    st.header("API密钥设置")
    openai_api_key = st.text_input("OpenAI API密钥", type="password")
    api_base = st.text_input("API服务商地址", value="https://api.openai.com/v1")
    
    st.header("模型设置")
    model_options = {
        "GPT-4o-mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o",
        "GPT-o1-mini": "o1-mini",
        "GPT-o1": "o1-preview",
        "GPT-o3-mini": "o3-mini",
    }
    
    selected_model = st.selectbox(
        "选择模型",
        options=list(model_options.keys()),
        index=0,
        help="选择要使用的OpenAI模型。GPT-4系列性能更好但成本更高。"
    )
    
    temperature = st.slider(
        "模型温度",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="较低的值使输出更确定性，较高的值使输出更创造性"
    )

# 主界面内容
if st.session_state.current_view == "input":
    # 输入字段
    company_name = st.text_input("请输入公司名称:")
    meeting_objective = st.text_input("请输入会议目标:")
    attendees = st.text_area("请输入参会者及其角色(每行一个):")
    meeting_duration = st.number_input("请输入会议时长(分钟):", min_value=15, max_value=180, value=60, step=15)
    focus_areas = st.text_input("请输入需要特别关注的领域或问题:")
    
    if st.button("准备会议", use_container_width=True):
        if not openai_api_key:
            st.warning("请先在侧边栏输入API密钥。")
        else:
            # 设置环境变量
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["OPENAI_API_BASE"] = api_base

            # 初始化LLM配置
            llm = LLM(
                model=model_options[selected_model],
                temperature=temperature,
                api_key=openai_api_key,
                api_base=api_base,
                system_prompt="你是一位专业的AI助手。请始终使用中文回答所有问题。确保输出的所有内容都是中文，包括标题、分析和建议。"
            )

            # 创建搜索工具
            wiki_tool = WikipediaToolWrapper()
            duckduckgo_tool = DuckDuckGoToolWrapper()

            # 定义AI助手
            context_analyzer = LoggingAgent(
                role='会议背景分析专家',
                goal='分析和总结会议的关键背景信息',
                backstory='你是一位擅长快速理解复杂商业背景并识别关键信息的专家。你会优先使用中文维基百科进行搜索和研究，并通过DuckDuckGo获取最新信息。',
                verbose=True,
                allow_delegation=False,
                llm=llm,
                tools=[wiki_tool, duckduckgo_tool]
            )

            industry_insights_generator = LoggingAgent(
                role='行业专家',
                goal='提供深入的行业分析并识别关键趋势',
                backstory='你是一位经验丰富的行业分析师，擅长发现新兴趋势和机会。你会结合维基百科的基础信息和DuckDuckGo的最新动态来分析行业情况。',
                verbose=True,
                allow_delegation=False,
                llm=llm,
                tools=[wiki_tool, duckduckgo_tool]
            )

            strategy_formulator = LoggingAgent(
                role='会议策略专家',
                goal='制定定制化的会议策略和详细议程',
                backstory='你是一位会议规划大师，以制定高效的策略和议程而闻名。',
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )

            executive_briefing_creator = LoggingAgent(
                role='沟通专家',
                goal='将信息综合成简明有力的简报',
                backstory='你是一位专业的沟通专家，擅长将复杂信息转化为清晰、可执行的见解。',
                verbose=True,
                allow_delegation=False,
                llm=llm,
            )

            # 定义任务
            context_analysis_task = Task(
                description=f"""
                分析与{company_name}会议相关的背景，考虑以下方面：
                1. 会议目标：{meeting_objective}
                2. 参会人员：{attendees}
                3. 会议时长：{meeting_duration}分钟
                4. 特别关注领域或问题：{focus_areas}

                深入研究{company_name}，包括：
                1. 最新新闻和新闻发布
                2. 主要产品或服务
                3. 主要竞争对手

                提供全面的调查结果总结，突出与会议背景最相关的信息。
                使用markdown格式输出，包含适当的标题和子标题。
                """,
                agent=context_analyzer,
                expected_output="一份详细的会议背景和公司背景分析，包括最新发展、财务表现以及与会议目标的相关性，使用markdown格式并包含标题和子标题。"
            )

            industry_analysis_task = Task(
                description=f"""
                基于{company_name}的背景分析和会议目标：{meeting_objective}，提供深入的行业分析：
                1. 识别行业关键趋势和发展
                2. 分析竞争格局
                3. 突出潜在机会和威胁
                4. 提供市场定位见解

                确保分析与会议目标和参会者角色相关。
                使用markdown格式输出，包含适当的标题和子标题。
                """,
                agent=industry_insights_generator,
                expected_output="一份全面的行业分析报告，包括趋势、竞争格局、机会、威胁以及与会议目标相关的见解，使用markdown格式并包含标题和子标题。"
            )

            strategy_development_task = Task(
                description=f"""
                根据背景分析和行业见解，为与{company_name}的{meeting_duration}分钟会议制定定制化会议策略和详细议程。包括：
                1. 带有明确目标的分时议程
                2. 每个议程项目的关键讨论要点
                3. 每个环节的建议发言人或主持人
                4. 潜在讨论话题和推动对话的问题
                5. 解决特定关注领域和问题的策略：{focus_areas}

                确保策略和议程与会议目标保持一致：{meeting_objective}
                使用markdown格式输出，包含适当的标题和子标题。
                """,
                agent=strategy_formulator,
                expected_output="一份详细的会议策略和分时议程，包括目标、关键讨论要点和解决特定关注领域的策略，使用markdown格式并包含标题和子标题。"
            )

            executive_brief_task = Task(
                description=f"""
                将所有收集的信息综合成一份全面而简明的{company_name}会议执行简报。创建以下内容：

                1. 详细的一页执行摘要，包括：
                   - 明确的会议目标陈述
                   - 主要参会者及其角色列表
                   - 关于{company_name}的关键背景要点和相关行业背景
                   - 与目标相一致的3-5个战略性会议目标
                   - 会议结构和将要讨论的关键主题概述

                2. 详细的关键讨论要点清单，每个要点都需要：
                   - 相关数据或统计
                   - 具体案例或案例研究
                   - 与公司当前情况或挑战的联系

                3. 预测并准备潜在问题：
                   - 根据参会者角色和会议目标列出可能的问题
                   - 为每个问题准备基于数据的回答
                   - 包含可能需要的任何支持信息或额外背景

                4. 战略建议和后续步骤：
                   - 基于分析提供3-5个可执行的建议
                   - 列出明确的实施或跟进步骤
                   - 建议关键行动的时间表或截止日期
                   - 识别潜在挑战或障碍并提出缓解策略

                确保简报全面而简明，具有高度可执行性，并与会议目标精确对齐：{meeting_objective}。文档结构应便于导航和会议期间快速参考。
                使用markdown格式输出，包含适当的标题和子标题。
                """,
                agent=executive_briefing_creator,
                expected_output="一份全面的执行简报，包括摘要、关键讨论要点、问答准备和战略建议，使用markdown格式，包含主标题(H1)、章节标题(H2)和小节标题(H3)。使用项目符号、编号列表和强调(粗体/斜体)突出关键信息。"
            )

            # 创建工作组
            meeting_prep_crew = LoggingCrew(
                agents=[context_analyzer, industry_insights_generator, strategy_formulator, executive_briefing_creator],
                tasks=[context_analysis_task, industry_analysis_task, strategy_development_task, executive_brief_task],
                verbose=True,
                process=Process.sequential
            )

            with st.spinner("AI助手团队正在协同工作..."):
                try:
                    # 执行分析
                    result = meeting_prep_crew.kickoff()
                    st.session_state.current_view = "context"  # 自动切换到背景分析视图
                    st.rerun()
                except Exception as e:
                    st.error(f"错误信息：{str(e)}")

else:
    # 显示分析结果
    if st.session_state.current_view == "context":
        st.header("🔍 会议背景分析")
        if st.session_state.expert_analysis['context']['result']:
            tab1, tab2 = st.tabs(["分析结果", "分析过程"])
            with tab1:
                st.markdown(st.session_state.expert_analysis['context']['result'])
            with tab2:
                for thought in st.session_state.expert_analysis['context']['thoughts']:
                    st.markdown(thought)
                for analysis in st.session_state.expert_analysis['context']['analysis']:
                    st.markdown("---")
                    st.markdown(analysis)
        else:
            st.info("暂无分析结果，请先进行会议准备。")
            
    elif st.session_state.current_view == "industry":
        st.header("📈 行业趋势分析")
        if st.session_state.expert_analysis['industry']['result']:
            tab1, tab2 = st.tabs(["分析结果", "分析过程"])
            with tab1:
                st.markdown(st.session_state.expert_analysis['industry']['result'])
            with tab2:
                for thought in st.session_state.expert_analysis['industry']['thoughts']:
                    st.markdown(thought)
                for analysis in st.session_state.expert_analysis['industry']['analysis']:
                    st.markdown("---")
                    st.markdown(analysis)
        else:
            st.info("暂无分析结果，请先进行会议准备。")
            
    elif st.session_state.current_view == "strategy":
        st.header("📋 会议策略")
        if st.session_state.expert_analysis['strategy']['result']:
            tab1, tab2 = st.tabs(["分析结果", "分析过程"])
            with tab1:
                st.markdown(st.session_state.expert_analysis['strategy']['result'])
            with tab2:
                for thought in st.session_state.expert_analysis['strategy']['thoughts']:
                    st.markdown(thought)
                for analysis in st.session_state.expert_analysis['strategy']['analysis']:
                    st.markdown("---")
                    st.markdown(analysis)
        else:
            st.info("暂无分析结果，请先进行会议准备。")
            
    elif st.session_state.current_view == "briefing":
        st.header("📑 执行简报")
        if st.session_state.expert_analysis['briefing']['result']:
            tab1, tab2 = st.tabs(["分析结果", "分析过程"])
            with tab1:
                st.markdown(st.session_state.expert_analysis['briefing']['result'])
            with tab2:
                for thought in st.session_state.expert_analysis['briefing']['thoughts']:
                    st.markdown(thought)
                for analysis in st.session_state.expert_analysis['briefing']['analysis']:
                    st.markdown("---")
                    st.markdown(analysis)
        else:
            st.info("暂无分析结果，请先进行会议准备。")

# 显示分析日志
if st.session_state.current_view != "input":
    with st.expander("📋 查看分析过程记录"):
        display_logs()

st.sidebar.markdown("""
## 如何使用本应用：
1. 在侧边栏输入您的API密钥
2. 提供所需的会议信息
3. 点击'准备会议'生成全面的会议准备材料包

AI助手将协同工作以:
- 分析会议背景和公司背景
- 提供行业见解和趋势
- 制定定制化会议策略和议程
- 创建包含关键讨论要点的执行简报

这个过程可能需要几分钟时间，请耐心等待！
""")