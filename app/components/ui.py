"""
UI组件模块
包含所有界面相关的组件和函数
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List

def init_session_state():
    """初始化session state"""
    # 导航和状态标志
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "input"
        st.session_state.has_data = False
    
    # 分析日志
    if 'analysis_log' not in st.session_state:
        st.session_state.analysis_log = []
    
    # 文档相关
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    
    if 'doc_processor' not in st.session_state:
        from app.core.document_processor import DocumentProcessor
        st.session_state.doc_processor = DocumentProcessor()
    
    if 'doc_analysis' not in st.session_state:
        st.session_state.doc_analysis = {}
    
    # 分析结果存储
    if 'expert_analysis' not in st.session_state:
        st.session_state.expert_analysis = {
            'context': {'result': '', 'thoughts': [], 'analysis': []},
            'industry': {'result': '', 'thoughts': [], 'analysis': []},
            'strategy': {'result': '', 'thoughts': [], 'analysis': []},
            'briefing': {'result': '', 'thoughts': [], 'analysis': []}
        }
    
    # LLM实例
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    
    # 日志容器
    if 'log_container' not in st.session_state:
        st.session_state.log_container = None

def create_navigation():
    """创建导航栏"""
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
    
    with nav_col1:
        if st.button("📝 输入信息", use_container_width=True):
            st.session_state.current_view = "input"
    with nav_col2:
        if st.button("🔍 会议背景", use_container_width=True):
            if st.session_state.expert_analysis['context']['result']:
                st.session_state.current_view = "context"
            else:
                st.warning("请先完成会议准备分析。")
    with nav_col3:
        if st.button("📈 行业趋势", use_container_width=True):
            if st.session_state.expert_analysis['industry']['result']:
                st.session_state.current_view = "industry"
            else:
                st.warning("请先完成会议准备分析。")
    with nav_col4:
        if st.button("📋 会议策略", use_container_width=True):
            if st.session_state.expert_analysis['strategy']['result']:
                st.session_state.current_view = "strategy"
            else:
                st.warning("请先完成会议准备分析。")
    with nav_col5:
        if st.button("📑 执行简报", use_container_width=True):
            if st.session_state.expert_analysis['briefing']['result']:
                st.session_state.current_view = "briefing"
            else:
                st.warning("请先完成会议准备分析。")

def display_sidebar():
    """显示侧边栏"""
    with st.sidebar:
        st.header("API密钥设置")
        openai_api_key = st.text_input("OpenAI API密钥", type="password")
        api_base = st.text_input("API服务商地址", value="https://api.openai.com/v1")
        
        st.header("模型设置")
        from ..config.settings import MODEL_OPTIONS, DEFAULT_TEMPERATURE
        
        selected_model = st.selectbox(
            "选择模型",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help="选择要使用的OpenAI模型。GPT-4系列性能更好但成本更高。"
        )
        
        temperature = st.slider(
            "模型温度",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="较低的值使输出更确定性，较高的值使输出更创造性"
        )
        
        st.header("功能设置")
        enable_wiki = st.checkbox("启用维基百科搜索", value=False, help="启用后，AI助手将使用维基百科搜索相关信息")
        
        return openai_api_key, api_base, selected_model, temperature, enable_wiki

def display_document_upload(doc_processor):
    """显示文档上传区域"""
    doc_col1, doc_col2 = st.columns([2, 1])
    
    with doc_col1:
        st.header("📄 上传相关文档")
        col1, col2 = st.columns([4, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "支持Word、PDF、Excel、图片等多种格式",
                accept_multiple_files=True,
                type=list(sum([exts for exts in doc_processor.supported_extensions.values()], []))
            )
        with col2:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.uploaded_docs = []
                st.rerun()
    
    with doc_col2:
        st.header("🔗 输入网页链接")
        url = st.text_input("输入相关网页URL")
        if url and st.button("获取内容", use_container_width=True):
            process_url(url, doc_processor)
    
    return uploaded_files

def process_url(url: str, doc_processor):
    """处理URL输入"""
    with st.spinner("正在获取网页内容..."):
        try:
            content = doc_processor.extract_from_url(url)
            doc_info = {
                'filename': url,
                'content': content,
                'type': 'url',
                'size': len(content),
                'timestamp': datetime.now().isoformat()
            }
            if not any(doc['filename'] == url for doc in st.session_state.uploaded_docs):
                st.session_state.uploaded_docs.append(doc_info)
                st.success("✅ 已获取网页内容")
            else:
                st.warning("⚠️ 该网页已添加")
        except Exception as e:
            st.error(f"❌ 获取失败：{str(e)}")

def display_uploaded_documents():
    """显示已上传的文档列表"""
    if st.session_state.uploaded_docs:
        st.subheader("📚 已上传的资料")
        for i, doc in enumerate(st.session_state.uploaded_docs):
            with st.expander(f"{'🌐' if doc['type'] == 'url' else '📄'} {doc['filename']}", expanded=False):
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.text("类型")
                    st.text("大小")
                    st.text("时间")
                with col2:
                    st.text(f": {doc['type']}")
                    st.text(f": {doc['size']} bytes")
                    st.text(f": {datetime.fromisoformat(doc['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                with col3:
                    if st.button("🗑️", key=f"del_{i}", help="删除此文档"):
                        st.session_state.uploaded_docs.pop(i)
                        st.rerun()
                
                preview = doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content']
                st.text_area(
                    "内容预览",
                    preview,
                    height=100,
                    key=f"doc_{i}_{doc['timestamp']}"
                )

def display_analysis_results():
    """显示分析结果"""
    if st.session_state.current_view == "context":
        display_tab_view('context')
    elif st.session_state.current_view == "industry":
        display_tab_view('industry')
    elif st.session_state.current_view == "strategy":
        display_tab_view('strategy')
    elif st.session_state.current_view == "briefing":
        display_tab_view('briefing')

def display_tab_view(view_type: str):
    """显示标签页视图"""
    tab1, tab2 = st.tabs(["分析结果", "分析过程"])
    with tab1:
        st.markdown(st.session_state.expert_analysis[view_type]['result'])
    with tab2:
        for thought in st.session_state.expert_analysis[view_type]['thoughts']:
            st.markdown(thought)
        for analysis in st.session_state.expert_analysis[view_type]['analysis']:
            st.markdown("---")
            st.markdown(analysis)

def display_logs():
    """显示分析日志"""
    if st.session_state.current_view != "input":
        with st.expander("📋 查看分析过程记录", expanded=True):
            log_container = st.empty()
            with log_container.container():
                for log in st.session_state.analysis_log:
                    st.markdown(log)
                st.markdown('<script>window.scrollTo(0,document.body.scrollHeight);</script>', unsafe_allow_html=True)

def display_help():
    """显示帮助信息"""
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