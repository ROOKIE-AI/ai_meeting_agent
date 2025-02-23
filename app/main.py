"""
主应用模块
整合所有模块，实现完整的应用功能
"""

import streamlit as st
from crewai import Task, LLM
from crewai.process import Process
import os

from app.core.document_processor import DocumentProcessor
from app.core.ai_agents import LoggingAgent, LoggingCrew
from app.utils.tools import WikipediaToolWrapper, validate_api_key
from app.components.ui import (
    init_session_state,
    create_navigation,
    display_sidebar,
    display_document_upload,
    display_uploaded_documents,
    display_analysis_results,
    display_logs,
    display_help
)
from app.config.settings import (
    MODEL_OPTIONS,
    SYSTEM_PROMPT,
    MIN_MEETING_DURATION,
    MAX_MEETING_DURATION,
    DEFAULT_MEETING_DURATION,
    MEETING_DURATION_STEP
)

def main():
    """主应用入口函数"""
    # 设置页面
    st.set_page_config(page_title="会议准备AI助手 📝", layout="wide")
    
    # 初始化session state（必须在使用任何session_state变量之前调用）
    init_session_state()
    
    st.title("会议准备AI助手 📝")
    
    # 创建导航栏
    create_navigation()
    st.divider()
    
    # 显示侧边栏
    openai_api_key, api_base, selected_model, temperature, enable_wiki = display_sidebar()
    
    # 显示帮助信息
    display_help()
    
    # 主界面内容
    if st.session_state.current_view == "input":
        # 文档处理
        if 'doc_processor' not in st.session_state:
            st.session_state.doc_processor = DocumentProcessor()
        
        uploaded_files = display_document_upload(st.session_state.doc_processor)
        
        # 处理上传的文档
        if uploaded_files:
            with st.spinner("正在处理文档..."):
                for file in uploaded_files:
                    if any(doc['filename'] == file.name for doc in st.session_state.uploaded_docs):
                        st.warning(f"⚠️ {file.name} 已存在，跳过处理")
                        continue
                    
                    if st.session_state.doc_processor.validate_file(file):
                        doc_info = st.session_state.doc_processor.process_file(file)
                        if 'error' not in doc_info:
                            st.session_state.uploaded_docs.append(doc_info)
                            st.success(f"✅ 已处理：{file.name}")
                        else:
                            st.error(f"❌ 处理失败 {file.name}：{doc_info['error']}")
                    else:
                        st.warning(f"⚠️ 不支持的格式：{file.name}")
        
        # 显示已上传的文档
        display_uploaded_documents()
        
        st.divider()
        
        # 输入字段
        company_name = st.text_input("请输入公司名称:")
        meeting_objective = st.text_input("请输入会议目标:")
        attendees = st.text_area("请输入参会者及其角色(每行一个):")
        meeting_duration = st.number_input(
            "请输入会议时长(分钟):",
            min_value=MIN_MEETING_DURATION,
            max_value=MAX_MEETING_DURATION,
            value=DEFAULT_MEETING_DURATION,
            step=MEETING_DURATION_STEP
        )
        focus_areas = st.text_input("请输入需要特别关注的领域或问题:")
        
        if st.button("准备会议", use_container_width=True):
            if not openai_api_key:
                st.warning("请先在侧边栏输入API密钥。")
            elif not validate_api_key(openai_api_key, api_base):
                st.error("API密钥格式无效！请检查密钥格式。")
            else:
                # 设置环境变量
                os.environ["OPENAI_API_KEY"] = openai_api_key
                os.environ["OPENAI_API_BASE"] = api_base

                try:
                    # 初始化LLM配置
                    llm = LLM(
                        model=MODEL_OPTIONS[selected_model],
                        temperature=temperature,
                        api_key=openai_api_key,
                        api_base=api_base,
                        system_prompt=SYSTEM_PROMPT
                    )
                    
                    # 测试LLM连接
                    test_response = llm.call("测试连接")
                    if not test_response:
                        st.error("API连接测试失败，请检查API密钥和服务地址是否正确。")
                    else:
                        st.session_state.llm = llm
                        
                        # 创建搜索工具
                        wiki_tool = WikipediaToolWrapper() if enable_wiki else None
                        
                        # 定义AI助手
                        context_analyzer = LoggingAgent(
                            role='会议背景分析专家',
                            goal='分析和总结会议的关键背景信息',
                            backstory='你是一位擅长快速理解复杂商业背景并识别关键信息的专家。' + 
                                     '你会分析提供的文档资料，并' + 
                                     ('使用中文维基百科进行搜索和研究。' if wiki_tool else '进行深入研究。'),
                            verbose=True,
                            allow_delegation=False,
                            llm=llm,
                            tools=[wiki_tool] if wiki_tool else []
                        )

                        industry_insights_generator = LoggingAgent(
                            role='行业专家',
                            goal='提供深入的行业分析并识别关键趋势',
                            backstory='你是一位经验丰富的行业分析师，擅长发现新兴趋势和机会。' + ('你会使用维基百科的信息来分析行业情况。' if wiki_tool else ''),
                            verbose=True,
                            allow_delegation=False,
                            llm=llm,
                            tools=[wiki_tool] if wiki_tool else []
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

                        # 准备文档上下文
                        docs_info = st.session_state.uploaded_docs if st.session_state.uploaded_docs else []
                        docs_context = "\n\n相关文档资料：\n" + "\n".join([
                            f"文档{i+1}. {doc['filename']}:\n{doc['content'][:1000]}..."
                            for i, doc in enumerate(docs_info)
                        ]) if docs_info else ""
                        
                        # 定义任务
                        tasks = [
                            Task(
                                description=f"""
                                分析与{company_name}会议相关的背景，考虑以下方面：
                                1. 会议目标：{meeting_objective}
                                2. 参会人员：{attendees}
                                3. 会议时长：{meeting_duration}分钟
                                4. 特别关注领域或问题：{focus_areas}
                                
                                {docs_context}

                                深入研究{company_name}，包括：
                                1. 最新新闻和新闻发布
                                2. 主要产品或服务
                                3. 主要竞争对手

                                提供全面的调查结果总结，突出与会议背景最相关的信息。
                                使用markdown格式输出，包含适当的标题和子标题。
                                """,
                                agent=context_analyzer,
                                expected_output="一份详细的会议背景和公司背景分析，包括最新发展、财务表现以及与会议目标的相关性，使用markdown格式并包含标题和子标题。"
                            ),
                            Task(
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
                            ),
                            Task(
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
                            ),
                            Task(
                                description=f"""
                                将所有收集的信息综合成一份全面而简明的{company_name}会议执行简报。创建以下内容：

                                1. 详细的一页执行摘要
                                2. 详细的关键讨论要点清单
                                3. 预测并准备潜在问题
                                4. 战略建议和后续步骤

                                确保简报全面而简明，具有高度可执行性，并与会议目标精确对齐：{meeting_objective}。
                                使用markdown格式输出，包含适当的标题和子标题。
                                """,
                                agent=executive_briefing_creator,
                                expected_output="一份全面的执行简报，包括摘要、关键讨论要点、问答准备和战略建议，使用markdown格式，包含主标题(H1)、章节标题(H2)和小节标题(H3)。使用项目符号、编号列表和强调(粗体/斜体)突出关键信息。"
                            )
                        ]

                        # 创建工作组
                        meeting_prep_crew = LoggingCrew(
                            agents=[context_analyzer, industry_insights_generator, strategy_formulator, executive_briefing_creator],
                            tasks=tasks,
                            verbose=True,
                            process=Process.sequential
                        )

                        with st.spinner("AI助手团队正在协同工作..."):
                            try:
                                # 创建日志显示区域
                                st.markdown("### 🔄 实时分析日志")
                                st.session_state.log_container = st.empty()
                                
                                # 执行分析
                                result = meeting_prep_crew.kickoff()
                                st.session_state.has_data = True
                                st.session_state.current_view = "context"
                                st.rerun()
                            except Exception as e:
                                st.error(f"错误信息：{str(e)}")

                except Exception as e:
                    st.error(f"API连接失败：{str(e)}")
                    st.info("请检查：\n1. API密钥是否正确\n2. API服务地址是否可访问\n3. 网络连接是否正常")
    else:
        # 显示分析结果
        display_analysis_results()
    
    # 显示分析日志
    display_logs()

if __name__ == "__main__":
    main() 