import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import networkx as nx
import os
import sys
import tempfile
import uuid
from datetime import datetime

# 添加当前目录到Python路径，便于导入其他分工模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from api_use import ChatAnalyzer  # 分工1

    DIVISION_1_AVAILABLE = True
    print("✅ 成功导入分工1模块")
except ImportError as e:
    DIVISION_1_AVAILABLE = False
    print(f"⚠️ 无法导入分工1模块: {e}")
    ChatAnalyzer = None

try:
    from Searcher import Searcher  # 分工3

    DIVISION_3_AVAILABLE = True
    print("✅ 成功导入分工3模块")
except ImportError as e:
    DIVISION_3_AVAILABLE = False
    print(f"⚠️ 无法导入分工3模块: {e}")
    Searcher = None

try:
    from graphs import TopicGraph  # 分工4

    DIVISION_4_AVAILABLE = True
    print("✅ 成功导入分工4模块")
except ImportError as e:
    DIVISION_4_AVAILABLE = False
    print(f"⚠️ 无法导入分工4模块: {e}")
    TopicGraph = None


class FrontendManager:
    def __init__(self):
        # 初始化session state
        self._init_session_state()

        # 创建必要的目录结构
        self._init_directory_structure()

        # 加载配置
        self.config = self._load_config()

        # 初始化各模块实例
        self.analyzer = None  # 分工1实例
        self.searcher = None  # 分工3实例
        self.topic_graph = None  # 分工4实例

        # 初始化模块
        self.init_modules()

    def _init_session_state(self):
        """初始化session state"""
        # 修改：添加各模块实例和配置的session state
        session_defaults = {
            'current_topic': None,
            'edit_mode': False,
            'uploaded_file': None,
            'analysis_data': None,
            'current_group': None,
            'topic_mapping': {},
            'data_file': None,
            'api_key': "",
            'base_url': "https://api-inference.modelscope.cn/v1/",
            'analyzer_instance': None,  # 存储分工1实例
            'searcher_instance': None,  # 存储分工3实例
            'topic_graph_instance': None,  # 存储分工4实例
            'modules_initialized': False,  # 模块是否已初始化
        }

        for key, default in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    def _init_directory_structure(self):
        """初始化项目目录结构"""
        # 修改：创建统一的目录结构
        directories = ['output', 'config', 'reports', 'temp']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

        # 创建默认配置文件
        config_file = 'config/api_config.json'
        if not os.path.exists(config_file):
            default_config = {
                "api_key": "",
                "base_url": "https://api-inference.modelscope.cn/v1/",
                "model": "Qwen/Qwen2.5-Coder-32B-Instruct"
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)

        # 创建API令牌文件（如果需要）
        token_file = 'config/api_token.txt'
        if not os.path.exists(token_file):
            with open(token_file, 'w', encoding='utf-8') as f:
                f.write("")

    def _load_config(self):
        """加载配置文件"""
        try:
            config_file = 'config/api_config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 更新session state中的API配置
                    st.session_state.api_key = config.get("api_key", "")
                    st.session_state.base_url = config.get("base_url", "https://api-inference.modelscope.cn/v1/")
                    return config
            return {}
        except Exception as e:
            st.error(f"加载配置文件失败: {str(e)}")
            return {}

    def init_modules(self):
        """初始化所有模块"""
        # 修改：直接初始化各分工模块实例

        # 初始化分工1模块（聊天记录分析）
        if DIVISION_1_AVAILABLE and st.session_state.api_key:
            try:
                st.session_state.analyzer_instance = ChatAnalyzer(
                    api_key=st.session_state.api_key,
                    base_url=st.session_state.base_url
                )
                self.analyzer = st.session_state.analyzer_instance
                print("✅ 分工1模块初始化成功")
            except Exception as e:
                print(f"❌ 分工1模块初始化失败: {e}")
                self.analyzer = None
        else:
            self.analyzer = None
            if not DIVISION_1_AVAILABLE:
                print("⚠️ 分工1模块不可用")
            else:
                print("⚠️ 分工1模块未初始化（缺少API密钥）")

        # 初始化分工3模块（智能搜索）
        if DIVISION_3_AVAILABLE:
            try:
                # 检查数据文件是否存在
                data_file = "output/search_data.json"
                token_file = "config/api_token.txt"

                # 如果数据文件不存在，创建一个空的
                if not os.path.exists(data_file):
                    with open(data_file, 'w', encoding='utf-8') as f:
                        json.dump({"chat_groups": []}, f)

                st.session_state.searcher_instance = Searcher(
                    data_file=data_file,
                    token_file=token_file
                )
                self.searcher = st.session_state.searcher_instance
                print("✅ 分工3模块初始化成功")
            except Exception as e:
                print(f"❌ 分工3模块初始化失败: {e}")
                self.searcher = None
        else:
            self.searcher = None
            print("⚠️ 分工3模块不可用")

        # 初始化分工4模块（话题图管理）
        if DIVISION_4_AVAILABLE:
            try:
                # 检查数据文件是否存在
                graph_file = "output/topic_graph_data.json"
                if not os.path.exists(graph_file):
                    with open(graph_file, 'w', encoding='utf-8') as f:
                        json.dump({"chat_groups": []}, f)

                st.session_state.topic_graph_instance = TopicGraph(graph_file)
                self.topic_graph = st.session_state.topic_graph_instance
                print("✅ 分工4模块初始化成功")
            except Exception as e:
                print(f"❌ 分工4模块初始化失败: {e}")
                self.topic_graph = None
        else:
            self.topic_graph = None
            print("⚠️ 分工4模块不可用")

        st.session_state.modules_initialized = True

    # ==================== 修改：文件上传和分析部分 ====================
    def handle_file_upload(self):
        """处理用户上传的聊天记录文件"""
        st.sidebar.markdown("### 📁 上传聊天记录")

        # API配置部分
        st.sidebar.markdown("### 🔑 API配置")
        col1, col2 = st.sidebar.columns([3, 1])

        with col1:
            api_key = st.text_input(
                "API密钥",
                value=st.session_state.api_key,
                type="password",
                help="输入Modelscope API密钥",
                key="api_key_input"
            )

        with col2:
            base_url = st.text_input(
                "API地址",
                value=st.session_state.base_url,
                help="API基础地址",
                key="base_url_input"
            )

        # 检查API配置是否变化
        if api_key != st.session_state.api_key or base_url != st.session_state.base_url:
            st.session_state.api_key = api_key
            st.session_state.base_url = base_url
            # 保存配置
            self._save_api_config(api_key, base_url)
            # 重新初始化模块
            self.init_modules()
            st.rerun()

        # 显示模块状态
        self._show_module_status_in_sidebar()

        uploaded_file = st.sidebar.file_uploader(
            "选择聊天记录文件",
            type=['txt', 'pdf', 'doc', 'docx'],
            help="支持TXT、PDF、DOC、DOCX格式的聊天记录文件",
            key="file_uploader"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

            # 显示文件信息
            file_details = {
                "文件名": uploaded_file.name,
                "文件大小": f"{uploaded_file.size / 1024:.1f} KB",
                "文件类型": uploaded_file.type.split('/')[-1].upper()
            }
            st.sidebar.write("文件信息:")
            for key, value in file_details.items():
                st.sidebar.write(f"- {key}: {value}")

            # 文件内容预览（仅文本文件）
            if uploaded_file.type.startswith('text/'):
                try:
                    content = uploaded_file.getvalue().decode('utf-8')
                    preview_lines = content.split('\n')[:5]
                    if any(line.strip() for line in preview_lines):
                        st.sidebar.write("**内容预览:**")
                        for line in preview_lines:
                            if line.strip():
                                st.sidebar.text(line[:50] + "..." if len(line) > 50 else line)
                except:
                    pass

            # 触发分析按钮
            if st.sidebar.button("🚀 开始分析", type="primary", key="analyze_button"):
                if not self.analyzer:
                    st.sidebar.error("请先设置正确的API密钥")
                else:
                    with st.spinner("正在分析聊天记录，请稍候..."):
                        # 直接调用分工1的分析方法
                        analysis_result = self._direct_analyze_file(uploaded_file)
                        if analysis_result:
                            st.session_state.analysis_data = analysis_result
                            # 默认选择第一个群聊
                            if analysis_result.get("chat_groups"):
                                st.session_state.current_group = analysis_result["chat_groups"][0]["group_id"]
                            # 构建话题映射
                            self._build_topic_mapping()
                            # 保存数据供其他模块使用
                            self._save_data_to_files()
                            # 重新初始化模块以加载新数据
                            self.init_modules()
                            st.sidebar.success("分析完成！")
                            st.rerun()
                        else:
                            st.sidebar.error("分析失败，请检查文件格式或重试")

        return uploaded_file

    def _show_module_status_in_sidebar(self):
        """在侧边栏显示模块状态"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 模块状态")

        # 分工1状态
        if self.analyzer:
            st.sidebar.success("✅ 分析模块: 已加载")
        else:
            st.sidebar.warning("⚠️ 分析模块: 未加载")

        # 分工3状态
        if self.searcher:
            st.sidebar.success("✅ 搜索模块: 已加载")
        else:
            st.sidebar.warning("⚠️ 搜索模块: 未加载")

        # 分工4状态
        if self.topic_graph:
            st.sidebar.success("✅ 话题图模块: 已加载")
        else:
            st.sidebar.warning("⚠️ 话题图模块: 未加载")

    def _save_api_config(self, api_key, base_url):
        """保存API配置"""
        config = {
            "api_key": api_key,
            "base_url": base_url,
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct"
        }

        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_file = os.path.join(config_dir, "api_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _direct_analyze_file(self, uploaded_file):
        """直接调用分工1进行文件分析"""
        try:
            # 创建临时文件
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_ext,
                    dir="temp"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # 使用分工1解析文件
                records = self.analyzer.parse_file(tmp_file_path)
                st.info(f"成功解析 {len(records)} 条记录")

                # 分析话题
                group_name = f"聊天记录_{os.path.splitext(uploaded_file.name)[0]}"
                description = f"来自文件: {uploaded_file.name}"

                # 如果有现有结构，可以合并
                existing_structure = None
                if st.session_state.analysis_data:
                    existing_structure = st.session_state.analysis_data

                # 生成话题结构
                result = self.analyzer.analyze_topics(
                    group_name=group_name,
                    chat_records=records,
                    existing_structure=existing_structure,
                    description=description
                )

                st.success("话题分析完成")
                return result

            finally:
                # 清理临时文件
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"分析过程出错: {str(e)}")
            return None

    # ==================== 修改：数据保存方法 ====================
    def _save_data_to_files(self):
        """保存分析数据到文件，供其他分工使用"""
        if not st.session_state.analysis_data:
            return

        # 创建output目录
        data_dir = "output"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 保存为统一的数据文件
        data_file = os.path.join(data_dir, "unified_data.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.analysis_data, f, ensure_ascii=False, indent=2)

        # 保存为搜索模块专用格式
        search_data_file = os.path.join(data_dir, "search_data.json")
        search_data = {
            "chat_groups": st.session_state.analysis_data.get("chat_groups", [])
        }
        with open(search_data_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, ensure_ascii=False, indent=2)

        # 保存为话题图模块专用格式
        graph_data_file = os.path.join(data_dir, "topic_graph_data.json")
        graph_data = {
            "chat_groups": st.session_state.analysis_data.get("chat_groups", [])
        }
        with open(graph_data_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 更新分工1实例的数据
        if self.analyzer:
            self.analyzer.chat_structure = st.session_state.analysis_data

        # 更新分工3实例的数据（重新加载）
        if self.searcher and os.path.exists(search_data_file):
            try:
                self.searcher.data_file = search_data_file
                self.searcher.data = self.searcher._load_data()
            except:
                pass

        # 更新分工4实例的数据（重新加载）
        if self.topic_graph and os.path.exists(graph_data_file):
            try:
                self.topic_graph.load_from_json(graph_data_file)
            except:
                pass

        st.session_state.data_file = data_file
        print("✅ 数据已保存到output目录")

    # ==================== 修改：搜索功能部分 ====================
    def call_search_api(self, query: str, search_type: str = "keyword"):
        """直接调用分工3的搜索功能"""
        if not self.searcher:
            st.warning("搜索模块未初始化")
            return {"keyword_results": [], "ai_recommendations": []}

        try:
            # 调用分工3的搜索方法
            search_results = self.searcher.search(
                query=query,
                use_ai=(search_type == "ai_semantic"),
                ai_max_results=10,
                group_name=None,
                topic_name=None,
                use_batch_mode=False,
                batch_size=20
            )

            return search_results

        except Exception as e:
            st.error(f"搜索失败: {str(e)}")
            return {"keyword_results": [], "ai_recommendations": [], "stats": {}}

    def _convert_search_results(self, search_results):
        """将搜索API返回的结果转换为前端格式"""
        converted = []

        # 处理关键词搜索结果
        if 'keyword_results' in search_results:
            for result in search_results['keyword_results']:
                converted.append({
                    'topic_id': result.get('topic_id', ''),
                    'topic_name': result.get('topic_name', ''),
                    'content': result.get('summaries', [''])[0] if result.get('summaries') else '',
                    'sender': result.get('group_info', {}).get('group_name', ''),
                    'score': result.get('search_score', 0) / 10.0,  # 归一化到0-1
                    'search_type': 'keyword',
                    'priority': result.get('priority', '中'),
                    'group_name': result.get('group_info', {}).get('group_name', '')
                })

        # 处理AI推荐结果
        if 'ai_recommendations' in search_results:
            for result in search_results['ai_recommendations']:
                topic_info = result.get('topic_info', {})
                converted.append({
                    'topic_id': topic_info.get('topic_id', ''),
                    'topic_name': topic_info.get('topic_name', ''),
                    'content': topic_info.get('summaries', [''])[0] if topic_info.get('summaries') else '',
                    'sender': topic_info.get('group_info', {}).get('group_name', ''),
                    'score': result.get('confidence', 0.5),
                    'search_type': 'ai',
                    'priority': topic_info.get('priority', '中'),
                    'group_name': topic_info.get('group_info', {}).get('group_name', '')
                })

        return converted

    # ==================== 修改：话题图功能部分 ====================
    def render_topic_graph(self, data):
        """渲染话题关系图谱"""
        st.title("🕸️ 话题关系图谱")

        # 添加控制面板
        col1, col2, col3 = st.columns([2, 1, 1])

        with col2:
            if st.button("🔄 刷新图结构", key="refresh_graph"):
                if self.topic_graph:
                    try:
                        self.topic_graph.load_from_json("output/topic_graph_data.json")
                        st.success("图结构已刷新")
                        st.rerun()
                    except Exception as e:
                        st.error(f"刷新失败: {str(e)}")

        with col3:
            if st.button("📊 显示统计", key="show_stats"):
                self._show_graph_statistics()

        if not data.get("chat_groups"):
            st.info("请先上传聊天记录文件并进行分析")
            return

        # 获取当前群聊的话题
        current_group_id = st.session_state.current_group
        topics = []
        group_name = ""

        if current_group_id:
            for group in data["chat_groups"]:
                if group["group_id"] == current_group_id:
                    topics = group.get("topics", [])
                    group_name = group['group_name']
                    break

        if not topics:
            # 如果没有选择特定群聊，使用所有话题
            topics = []
            for group in data["chat_groups"]:
                topics.extend(group.get("topics", []))
            group_name = "所有群聊"

        if not topics:
            st.warning("没有找到话题数据")
            return

        st.caption(f"当前显示: {group_name} ({len(topics)}个话题)")

        # 使用分工4的话题图功能（如果可用）
        if self.topic_graph:
            self._render_advanced_topic_graph(topics, group_name)
        else:
            # 使用基础可视化
            self._render_basic_topic_graph(topics, group_name)

    def _render_advanced_topic_graph(self, topics, group_name):
        """使用分工4模块渲染高级话题图"""
        # 显示图结构统计
        with st.expander("📈 图结构统计", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("节点数", len(self.topic_graph.graph))
            with col2:
                total_connections = sum(len(conn) for conn in self.topic_graph.graph.values())
                st.metric("连接数", total_connections)
            with col3:
                st.metric("话题总数", len(topics))

            # 显示话题连接详情
            if st.checkbox("显示详细连接", key="show_connections"):
                st.write("**话题连接关系:**")
                connection_count = 0
                for topic in topics:
                    topic_id = topic['topic_id']
                    connections = self.topic_graph.graph.get(topic_id, [])
                    if connections:
                        connected_names = []
                        for conn_id in connections:
                            conn_name = self.topic_graph.get_topic_name_by_id(conn_id)
                            if conn_name:
                                connected_names.append(conn_name)

                        if connected_names:
                            st.write(f"- **{topic['topic_name']}** → {', '.join(connected_names)}")
                            connection_count += 1

                if connection_count == 0:
                    st.info("暂无连接关系")

        # 使用原有的networkx可视化
        self._render_basic_topic_graph(topics, group_name)

    def _render_basic_topic_graph(self, topics, group_name):
        """渲染基础话题图（原有功能）"""
        # 创建网络图
        G = nx.Graph()

        # 添加节点
        for topic in topics:
            priority_value = {"高": 100, "中": 70, "低": 40}.get(topic.get("priority", "中"), 50)
            G.add_node(topic['topic_id'],
                       label=topic['topic_name'],
                       size=priority_value,
                       summary=topic.get('summaries', [''])[0],
                       priority=topic.get('priority', '中'))

        # 添加边（基于related_topics）
        edge_count = 0
        for topic in topics:
            topic_id = topic['topic_id']
            for related_topic_name in topic.get("related_topics", []):
                # 查找相关话题的ID
                related_topic_id = None
                for t in topics:
                    if t['topic_name'] == related_topic_name:
                        related_topic_id = t['topic_id']
                        break

                if related_topic_id and related_topic_id != topic_id:
                    # 计算关系强度
                    strength = 0.5
                    if topic.get("priority") == "高":
                        strength += 0.2
                    if related_topic_name in topic.get("summaries", ["", ""])[0]:
                        strength += 0.3

                    if related_topic_id not in G[topic_id]:
                        G.add_edge(topic_id, related_topic_id,
                                   weight=strength,
                                   description=f"{topic['topic_name']} ↔ {related_topic_name}")
                        edge_count += 1

        if len(G.nodes()) == 0:
            st.warning("没有可显示的话题数据")
            return

        # 使用Plotly可视化
        pos = nx.spring_layout(G, k=1, iterations=50)

        edge_x = []
        edge_y = []
        edge_text = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2].get('description', f"关联强度: {edge[2].get('weight', 0):.2f}"))

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_info = G.nodes[node]
            display_summary = node_info['summary'][:50] + "..." if len(node_info['summary']) > 50 else node_info[
                'summary']
            node_text.append(f"{node_info['label']}<br>优先级: {node_info['priority']}<br>摘要: {display_summary}")
            node_size.append(node_info['size'])

            # 根据优先级设置颜色
            priority_color = {
                "高": '#FF6B6B',
                "中": '#4ECDC4',
                "低": '#45B7D1'
            }
            node_color.append(priority_color.get(node_info['priority'], '#45B7D1'))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[node]['label'] for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='darkblue')
            ),
            hovertext=node_text
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'话题关系网络 - {group_name}',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="节点大小表示优先级，颜色表示优先级等级（红-高，青-中，蓝-低）",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        st.plotly_chart(fig, use_container_width=True)

        # 图例说明
        st.info("💡 **图解**: 节点大小表示话题优先级，连线表示话题之间的关联关系，连线越粗表示关系强度越大")

    def _show_graph_statistics(self):
        """显示图结构统计信息"""
        if not self.topic_graph:
            st.warning("话题图模块未初始化")
            return

        try:
            # 显示详细统计
            st.subheader("📊 话题图详细统计")

            # 计算各种统计信息
            total_groups = len(self.topic_graph.chat_groups)
            total_topics = sum(len(group.get('topics', [])) for group in self.topic_graph.chat_groups)
            total_connections = sum(len(conn) for conn in self.topic_graph.graph.values())

            # 计算连接密度
            if total_topics > 1:
                max_possible_connections = total_topics * (total_topics - 1) / 2
                connection_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
            else:
                connection_density = 0

            # 显示统计信息
            metrics_data = [
                ("群聊数量", total_groups),
                ("话题总数", total_topics),
                ("连接总数", total_connections),
                ("连接密度", f"{connection_density:.2%}")
            ]

            cols = st.columns(4)
            for i, (label, value) in enumerate(metrics_data):
                with cols[i]:
                    st.metric(label, value)

            # 显示话题优先级分布
            priority_count = {"高": 0, "中": 0, "低": 0}
            for group in self.topic_graph.chat_groups:
                for topic in group.get('topics', []):
                    priority = topic.get('priority', '中')
                    priority_count[priority] = priority_count.get(priority, 0) + 1

            st.subheader("📋 话题优先级分布")
            fig = go.Figure(data=[go.Pie(
                labels=list(priority_count.keys()),
                values=list(priority_count.values()),
                hole=.3,
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            )])
            fig.update_layout(
                title="话题优先级分布",
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"显示统计信息失败: {str(e)}")

    def update_topic(self, topic_id: str, new_summary: str):
        """更新话题信息"""
        if not st.session_state.analysis_data:
            return False

        try:
            # 查找并更新话题
            for group in st.session_state.analysis_data.get("chat_groups", []):
                for topic in group.get("topics", []):
                    if topic['topic_id'] == topic_id:
                        # 更新摘要
                        if 'summaries' not in topic:
                            topic['summaries'] = []

                        if topic['summaries']:
                            # 更新第一个摘要
                            topic['summaries'][0] = new_summary
                        else:
                            topic['summaries'] = [new_summary]

                        # 如果分工4模块存在，更新话题图数据
                        if self.topic_graph:
                            self._update_topic_in_graph(topic_id, new_summary)

                        # 保存更新后的数据
                        self._save_data_to_files()

                        return True

            return False

        except Exception as e:
            st.error(f"更新话题失败: {str(e)}")
            return False

    def _update_topic_in_graph(self, topic_id, new_summary):
        """更新话题图数据"""
        try:
            # 重新加载数据
            graph_file = "output/topic_graph_data.json"
            self.topic_graph.load_from_json(graph_file)
        except Exception as e:
            print(f"更新话题图失败: {e}")

    def generate_topic_report(self, topic_id):
        """调用分工1生成报告"""
        if not self.analyzer:
            st.error("分析模块未初始化，无法生成报告")
            return None

        try:
            # 确保分析器中有当前数据
            if not self.analyzer.chat_structure:
                self.analyzer.chat_structure = st.session_state.analysis_data

            # 生成详细报告
            report_content = self.analyzer.generate_report(
                topic_id=topic_id,
                report_type="detailed"
            )

            return report_content

        except Exception as e:
            st.error(f"生成报告失败: {str(e)}")
            return None

    # ==================== 保留原有方法（稍作修改） ====================
    def load_data(self):
        """加载分析数据"""
        if st.session_state.analysis_data is not None:
            return st.session_state.analysis_data

        # 如果没有分析数据，显示空状态
        return {
            "analysis_info": {
                "total_messages": 0,
                "participants": 0,
                "core_topics": [],
                "main_achievements": [],
                "pending_items": []
            },
            "chat_groups": []
        }

    def _build_topic_mapping(self):
        """构建话题ID到话题名称的映射关系"""
        topic_mapping = {}
        if st.session_state.analysis_data:
            for group in st.session_state.analysis_data.get("chat_groups", []):
                for topic in group.get("topics", []):
                    topic_mapping[topic["topic_id"]] = {
                        "name": topic["topic_name"],
                        "group_id": group["group_id"],
                        "group_name": group["group_name"]
                    }
        st.session_state.topic_mapping = topic_mapping

    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("💬 群聊分析系统")
        st.sidebar.markdown("---")

        # 文件上传部分
        uploaded_file = self.handle_file_upload()

        st.sidebar.markdown("---")

        # 数据源状态显示
        if st.session_state.analysis_data is not None:
            groups = st.session_state.analysis_data.get("chat_groups", [])
            if groups:
                st.sidebar.success(f"✅ 已分析 {len(groups)} 个群聊")
            else:
                st.sidebar.success("✅ 使用分析结果数据")
        elif st.session_state.uploaded_file is not None:
            st.sidebar.warning("📁 文件已上传，等待分析")
        else:
            st.sidebar.info("📋 请上传聊天记录文件进行分析")

        # 群聊选择（如果有多个群聊）
        data = self.load_data()
        groups = data.get("chat_groups", [])
        if len(groups) > 1:
            st.sidebar.markdown("### 👥 选择群聊")
            group_options = [f"{group['group_name']} ({len(group.get('topics', []))}个话题)" for group in groups]
            selected_group_index = st.sidebar.selectbox(
                "选择要分析的群聊",
                range(len(groups)),
                format_func=lambda x: group_options[x],
                key="group_selector"
            )
            if selected_group_index is not None:
                st.session_state.current_group = groups[selected_group_index]["group_id"]

        # 筛选选项
        st.sidebar.markdown("### 🔍 筛选选项")
        priority_filter = st.sidebar.multiselect(
            "优先级筛选",
            ["高", "中", "低"],
            default=["高", "中", "低"],
            key="priority_filter"
        )

        # 导航
        st.sidebar.markdown("### 🧭 导航")
        page = st.sidebar.radio("选择页面", [
            "📊 分析概览",
            "🗂️ 话题浏览",
            "🕸️ 话题图谱",
            "🔍 智能搜索"
        ], key="page_navigation")

        # 重置按钮
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 重置所有数据", key="reset_all"):
            st.session_state.uploaded_file = None
            st.session_state.analysis_data = None
            st.session_state.current_topic = None
            st.session_state.edit_mode = False
            st.session_state.current_group = None
            st.session_state.topic_mapping = {}
            st.session_state.data_file = None
            st.rerun()

        return page, priority_filter

    def render_overview(self, data):
        """渲染分析概览页面"""
        st.title("📊 群聊分析概览")

        # 显示数据来源状态
        if st.session_state.analysis_data is not None:
            groups = data.get("chat_groups", [])
            if groups:
                st.success(f"✅ 已成功分析 {len(groups)} 个群聊")
            else:
                st.success("✅ 使用分析结果数据")
        else:
            st.info("📋 请上传聊天记录文件开始分析")

        if not data.get("chat_groups"):
            return

        # 计算统计信息
        total_messages = 0
        total_topics = 0
        participants_set = set()
        all_topics = []

        for group in data["chat_groups"]:
            for topic in group.get("topics", []):
                total_topics += 1
                # 从相关记录中提取参与者
                for record in topic.get("related_records", []):
                    if isinstance(record, str):
                        if "：" in record:
                            parts = record.split("：", 1)
                            if parts and parts[0].strip():
                                participants_set.add(parts[0].strip())
                        elif ":" in record:
                            parts = record.split(":", 1)
                            if parts and parts[0].strip():
                                participants_set.add(parts[0].strip())
                total_messages += len(topic.get("related_records", []))
                all_topics.append(topic['topic_name'])

        # 关键指标卡片
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("总消息数", f"{total_messages} 条")
        with col2:
            st.metric("参与人数", f"{len(participants_set)} 人")
        with col3:
            st.metric("总话题数", f"{total_topics} 个")

        st.markdown("---")

        # 群聊概览
        st.subheader("👥 群聊概览")
        for group in data["chat_groups"]:
            with st.expander(f"{group['group_name']} ({len(group.get('topics', []))}个话题)"):
                st.write(f"**描述**: {group.get('description', '暂无描述')}")
                st.write(f"**群聊ID**: {group['group_id']}")

                # 话题优先级统计
                priority_count = {"高": 0, "中": 0, "低": 0}
                for topic in group.get("topics", []):
                    priority = topic.get("priority", "中")
                    priority_count[priority] = priority_count.get(priority, 0) + 1

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("高优先级", priority_count["高"])
                with col2:
                    st.metric("中优先级", priority_count["中"])
                with col3:
                    st.metric("低优先级", priority_count["低"])

        # 分析结果
        if data.get("analysis_info", {}).get("main_achievements"):
            st.markdown("---")
            st.subheader("✅ 主要成果")
            for achievement in data["analysis_info"]["main_achievements"]:
                st.write(f"• {achievement}")

        if data.get("analysis_info", {}).get("pending_items"):
            st.markdown("---")
            st.subheader("⏳ 待决事项")
            for pending in data["analysis_info"]["pending_items"]:
                st.write(f"• {pending}")

        # 话题优先级分布
        if total_topics > 0:
            st.markdown("---")
            st.subheader("📊 话题优先级分布")

            priority_counts = {"高": 0, "中": 0, "低": 0}
            for group in data["chat_groups"]:
                for topic in group.get("topics", []):
                    priority = topic.get("priority", "中")
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1

            fig = go.Figure(data=[go.Pie(
                labels=list(priority_counts.keys()),
                values=list(priority_counts.values()),
                hole=.3,
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            )])
            fig.update_layout(
                title="话题优先级分布",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_topics_browse(self, data, priority_filter):
        """渲染话题浏览页面"""
        st.title("🗂️ 话题浏览")

        if not data.get("chat_groups"):
            st.info("请先上传聊天记录文件进行分析")
            return

        # 获取当前选择的群聊话题
        current_group_id = st.session_state.current_group
        current_topics = []

        if current_group_id:
            for group in data["chat_groups"]:
                if group["group_id"] == current_group_id:
                    current_topics = group.get("topics", [])
                    st.caption(f"当前群聊: {group['group_name']} ({len(current_topics)}个话题)")
                    break

        if not current_topics:
            # 如果没有选择特定群聊或群聊没有话题，显示所有话题
            current_topics = []
            for group in data["chat_groups"]:
                current_topics.extend(group.get("topics", []))
            if current_topics:
                st.caption(f"显示所有群聊的话题 ({len(current_topics)}个)")

        if not current_topics:
            st.info("没有找到任何话题")
            return

        # 话题筛选和排序
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input("搜索话题", placeholder="输入关键词搜索...", key="topic_search")

        with col2:
            sort_by = st.selectbox("排序方式", ["优先级降序", "相关记录数降序", "名称排序"], key="topic_sort")

        # 过滤话题
        filtered_topics = []
        for topic in current_topics:
            # 优先级筛选
            topic_priority = topic.get("priority", "中")
            if priority_filter and topic_priority not in priority_filter:
                continue

            # 关键词筛选
            if search_term:
                search_lower = search_term.lower()
                name_match = search_lower in topic['topic_name'].lower()
                summary_match = False
                for summary in topic.get("summaries", []):
                    if search_lower in summary.lower():
                        summary_match = True
                        break
                if not (name_match or summary_match):
                    continue

            filtered_topics.append(topic)

        if not filtered_topics:
            st.warning("没有找到符合条件的的话题")
            return

        # 排序
        if sort_by == "优先级降序":
            priority_order = {"高": 3, "中": 2, "低": 1}
            filtered_topics.sort(key=lambda x: priority_order.get(x.get("priority", "中"), 0), reverse=True)
        elif sort_by == "相关记录数降序":
            filtered_topics.sort(key=lambda x: len(x.get("related_records", [])), reverse=True)
        elif sort_by == "名称排序":
            filtered_topics.sort(key=lambda x: x['topic_name'])

        # 显示统计信息
        priority_count = {"高": 0, "中": 0, "低": 0}
        for topic in filtered_topics:
            priority = topic.get("priority", "中")
            priority_count[priority] = priority_count.get(priority, 0) + 1

        st.write(f"显示 {len(filtered_topics)} 个话题")

        # 显示话题列表
        for i, topic in enumerate(filtered_topics):
            self._render_topic_card(topic, i)

    def _render_topic_card(self, topic, index):
        """渲染单个话题卡片"""
        # 根据优先级设置颜色
        priority_color = {
            "高": "#FF6B6B",  # 红色
            "中": "#4ECDC4",  # 青色
            "低": "#45B7D1"  # 蓝色
        }
        color = priority_color.get(topic.get("priority", "中"), "#45B7D1")

        with st.expander(
                f"🔸 {topic['topic_name']} (优先级: {topic.get('priority', '中')}, 相关记录: {len(topic.get('related_records', []))})",
                expanded=index == 0):

            col1, col2 = st.columns([3, 1])

            with col1:
                # 显示摘要
                if topic.get("summaries"):
                    st.write(f"**📝 摘要**: {topic['summaries'][0]}")

                # 相关话题链接
                if topic.get("related_topics"):
                    st.write(f"**🔗 相关话题**: {', '.join(topic['related_topics'][:3])}")
                    if len(topic['related_topics']) > 3:
                        st.caption(f"等{len(topic['related_topics'])}个相关话题")

            with col2:
                if st.button("查看详情", key=f"view_{topic['topic_id']}"):
                    st.session_state.current_topic = topic['topic_id']
                    st.session_state.edit_mode = False
                    st.rerun()

                if st.button("编辑", key=f"edit_{topic['topic_id']}"):
                    st.session_state.current_topic = topic['topic_id']
                    st.session_state.edit_mode = True
                    st.rerun()

            # 如果当前话题被选中，显示详细信息
            if st.session_state.current_topic == topic['topic_id']:
                self._render_topic_detail(topic)

    def _render_topic_detail(self, topic):
        """渲染话题详细信息"""
        st.markdown("---")
        st.subheader(f"💬 {topic['topic_name']} 的详细记录")

        if st.session_state.edit_mode:
            # 编辑模式
            current_summary = topic['summaries'][0] if topic.get('summaries') else ""
            new_summary = st.text_area("话题摘要", value=current_summary, height=100,
                                       key=f"edit_summary_{topic['topic_id']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💾 保存修改", key=f"save_{topic['topic_id']}"):
                    if self.update_topic(topic['topic_id'], new_summary):
                        st.success("保存成功！")
                        st.session_state.edit_mode = False
                        st.rerun()
            with col2:
                if st.button("📄 生成报告", key=f"report_{topic['topic_id']}"):
                    report_content = self.generate_topic_report(topic['topic_id'])
                    if report_content:
                        with st.expander("📋 话题分析报告", expanded=True):
                            st.markdown(report_content)

                            # 提供下载按钮
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            report_text = f"""# 话题分析报告
## 话题名称: {topic['topic_name']}
## 话题ID: {topic['topic_id']}
## 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{report_content}
"""
                            st.download_button(
                                label="📥 下载报告",
                                data=report_text,
                                file_name=f"topic_report_{topic['topic_id']}_{timestamp}.md",
                                mime="text/markdown"
                            )
            with col3:
                if st.button("❌ 取消", key=f"cancel_{topic['topic_id']}"):
                    st.session_state.edit_mode = False
                    st.rerun()
        else:
            # 查看模式
            # 显示所有摘要
            if topic.get("summaries"):
                st.write("**话题摘要:**")
                for i, summary in enumerate(topic['summaries'], 1):
                    st.write(f"{i}. {summary}")

            # 显示相关聊天记录
            if topic.get("related_records"):
                st.write("**相关聊天记录:**")
                for record in topic.get("related_records", []):
                    if isinstance(record, str):
                        if "：" in record:
                            parts = record.split("：", 1)
                            if len(parts) == 2:
                                st.write(f"**{parts[0]}**: {parts[1]}")
                            else:
                                st.write(f"{record}")
                        elif ":" in record:
                            parts = record.split(":", 1)
                            if len(parts) == 2:
                                st.write(f"**{parts[0]}**: {parts[1]}")
                            else:
                                st.write(f"{record}")
                        else:
                            st.write(f"{record}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("返回列表", key=f"back_{topic['topic_id']}"):
                    st.session_state.current_topic = None
                    st.rerun()
            with col2:
                if st.button("编辑话题", key=f"edit_btn_{topic['topic_id']}"):
                    st.session_state.edit_mode = True
                    st.rerun()
            with col3:
                if st.button("生成报告", key=f"gen_report_{topic['topic_id']}"):
                    report_content = self.generate_topic_report(topic['topic_id'])
                    if report_content:
                        with st.expander("📋 话题分析报告", expanded=True):
                            st.markdown(report_content)

                            # 提供下载按钮
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            report_text = f"""# 话题分析报告
## 话题名称: {topic['topic_name']}
## 话题ID: {topic['topic_id']}
## 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{report_content}
"""
                            st.download_button(
                                label="📥 下载报告",
                                data=report_text,
                                file_name=f"topic_report_{topic['topic_id']}_{timestamp}.md",
                                mime="text/markdown"
                            )

    def render_search(self, data):
        """渲染智能搜索页面"""
        st.title("🔍 智能搜索")

        if not data.get("chat_groups"):
            st.info("请先上传聊天记录文件进行分析")
            return

        # 搜索输入
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("输入搜索内容", placeholder="输入关键词或完整句子...", key="search_input")
        with col2:
            search_type = st.selectbox("搜索类型", ["关键词", "语义"], key="search_type")
        with col3:
            st.write("")
            st.write("")
            search_button = st.button("开始搜索", type="primary", key="search_button")

        if search_button and search_query:
            st.write(f"正在搜索: `{search_query}`")

            # 调用分工3的搜索API
            with st.spinner("正在搜索..."):
                search_type_param = "keyword" if search_type == "关键词" else "ai_semantic"
                search_results = self.call_search_api(search_query, search_type_param)

            if search_results and "keyword_results" in search_results:
                # 转换搜索结果
                formatted_results = self._convert_search_results(search_results)

                if formatted_results:
                    # 按话题分组显示结果
                    results_by_topic = {}
                    for result in formatted_results:
                        topic_id = result['topic_id']
                        if topic_id not in results_by_topic:
                            results_by_topic[topic_id] = {
                                'topic_name': result['topic_name'],
                                'topic_id': topic_id,
                                'group_name': result.get('group_name', ''),
                                'priority': result.get('priority', '中'),
                                'results': [],
                                'max_score': result['score']
                            }
                        results_by_topic[topic_id]['results'].append(result)
                        if result['score'] > results_by_topic[topic_id]['max_score']:
                            results_by_topic[topic_id]['max_score'] = result['score']

                    # 按最高分排序
                    sorted_topics = sorted(results_by_topic.items(),
                                           key=lambda x: x[1]['max_score'],
                                           reverse=True)

                    st.success(f"找到 {len(formatted_results)} 条相关结果，分布在 {len(sorted_topics)} 个话题中")

                    for topic_id, topic_data in sorted_topics:
                        with st.expander(
                                f"📌 {topic_data['topic_name']} (相关度: {topic_data['max_score']:.2f}, {len(topic_data['results'])}条结果)"):
                            # 显示话题基本信息
                            st.write(f"**群聊**: {topic_data['group_name']}")
                            st.write(f"**优先级**: {topic_data['priority']}")

                            # 显示搜索结果
                            for i, result in enumerate(topic_data['results']):
                                st.write(f"**匹配内容**: {result['content']}")
                                st.write(f"**搜索类型**: {'关键词匹配' if result['search_type'] == 'keyword' else '语义匹配'}")
                                st.write(f"**相关度**: {result['score']:.2f}")

                                # 提供跳转到话题的链接
                                if st.button(f"查看该话题详情", key=f"goto_{topic_id}_{i}"):
                                    st.session_state.current_topic = topic_id
                                    st.rerun()

                                if i < len(topic_data['results']) - 1:
                                    st.divider()
                else:
                    st.warning("没有找到相关结果")
            else:
                st.warning("搜索服务返回空结果或发生错误")

    def run(self):
        """运行主应用"""
        # 加载数据
        data = self.load_data()

        # 渲染侧边栏并获取当前页面
        page, priority_filter = self.render_sidebar()

        # 根据选择渲染不同页面
        if page == "📊 分析概览":
            self.render_overview(data)
        elif page == "🗂️ 话题浏览":
            self.render_topics_browse(data, priority_filter)
        elif page == "🕸️ 话题图谱":
            self.render_topic_graph(data)
        elif page == "🔍 智能搜索":
            self.render_search(data)

if __name__ == "__main__":
    # 初始化页面配置
    st.set_page_config(
        page_title="群聊分析系统",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 显示标题和说明
    st.title("💬 群聊分析系统")
    st.markdown("""
    ### 使用步骤
    1. 在左侧输入API密钥（使用Modelscope API）
    2. 上传聊天记录文件（支持TXT、PDF、DOC、DOCX格式）
    3. 点击"开始分析"按钮进行分析
    4. 使用不同页面查看分析结果

    ### 模块状态
    页面左侧会显示各模块的加载状态，确保所有模块正常加载以获得完整功能。
    """)

    # 创建前端管理器实例并运行
    try:
        frontend = FrontendManager()
        frontend.run()
    except Exception as e:
        st.error(f"系统初始化失败: {str(e)}")
        st.info("请确保所有依赖模块（analyzer.py, searcher.py, topic_graph.py）在当前目录下")