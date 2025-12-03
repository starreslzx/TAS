import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import networkx as nx
import requests
import os

st.set_page_config(
    page_title="群聊分析系统",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


class FrontendManager:
    def __init__(self):
        if 'current_topic' not in st.session_state:
            st.session_state.current_topic = None
        if 'edit_mode' not in st.session_state:
            st.session_state.edit_mode = False
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'current_group' not in st.session_state:
            st.session_state.current_group = None
        if 'topic_mapping' not in st.session_state:
            st.session_state.topic_mapping = {}
        if 'data_file' not in st.session_state:
            st.session_state.data_file = None

    def handle_file_upload(self):
        """处理聊天记录文件"""
        st.sidebar.markdown("### 📁 上传聊天记录")

        uploaded_file = st.sidebar.file_uploader(
            "选择聊天记录文件",
            type=['txt', 'pdf', 'doc', 'docx'],
            help="支持TXT、PDF、DOC、DOCX格式的聊天记录文件"
        )

        if uploaded_file is not None:
            # 保存文件信息
            st.session_state.uploaded_file = uploaded_file

            # 显示文件信息
            file_details = {
                "文件名": uploaded_file.name,
                "文件大小": f"{uploaded_file.size / 1024:.1f} KB",
                "文件类型": uploaded_file.type
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
                    st.sidebar.warning("无法预览文件内容")

            # 分析
            if st.sidebar.button("🚀 开始分析", type="primary"):
                with st.spinner("正在分析聊天记录，请稍候..."):
                    # 调用分工1的分析API
                    analysis_result = self.call_analysis_api(uploaded_file)
                    if analysis_result:
                        st.session_state.analysis_data = analysis_result
                        # 默认选择第一个群聊
                        if analysis_result.get("chat_groups"):
                            st.session_state.current_group = analysis_result["chat_groups"][0]["group_id"]
                        # 构建话题映射
                        self._build_topic_mapping()
                        # 保存数据文件供分工3和分工4使用
                        self._save_data_to_file()
                        st.sidebar.success("分析完成！")
                        st.rerun()
                    else:
                        st.sidebar.error("分析失败，请检查文件格式或重试")

        return uploaded_file

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

    def _save_data_to_file(self):
        if not st.session_state.analysis_data:
            return

        # 保存到固定位置的文件
        data_dir = "output"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 保存原始数据供分工4使用
        raw_data_file = os.path.join(data_dir, "chat_topics_raw.json")
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.analysis_data, f, ensure_ascii=False, indent=2)

        # 保存搜索格式数据供分工3使用
        search_data_file = os.path.join(data_dir, "search_data.json")
        search_data = {
            "chat_groups": st.session_state.analysis_data.get("chat_groups", [])
        }
        with open(search_data_file, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, ensure_ascii=False, indent=2)

        # 保存更新文件供分工4读取
        updated_file = os.path.join(data_dir, "topics_data_updated.json")
        with open(updated_file, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.analysis_data, f, ensure_ascii=False, indent=2)

        st.session_state.data_file = search_data_file
        st.success(f"数据已保存到: {search_data_file}")

    def call_analysis_api(self, uploaded_file):
        """调用分工1的分析API处理上传的文件"""
        try:
            # 准备API请求
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            headers = {'Accept': 'application/json'}

            # 调用分工1的API
            response = requests.post(
                'http://localhost:8000/api/analyze-chat',
                files=files,
                headers=headers,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                # 检查返回的数据格式
                if isinstance(result, dict) and 'chat_groups' in result:
                    # 直接使用分工1返回的数据格式
                    return result
                else:
                    # 如果分工1返回了其他格式，尝试转换
                    return self._convert_to_frontend_format(result)
            else:
                st.error(f"分析服务错误: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            st.error("分析请求超时，请稍后重试或检查服务状态")
            return None
        except requests.exceptions.ConnectionError:
            st.error("无法连接到分析服务，请确保分工1的服务正在运行")
            return None
        except Exception as e:
            st.error(f"调用分析服务失败: {str(e)}")
            return None

    def _convert_to_frontend_format(self, backend_data):
        """分工1的数据格式转换"""
        if not backend_data:
            return None

        if isinstance(backend_data, dict) and 'chat_groups' in backend_data:
            return backend_data

        # 否则创建一个默认格式
        return {
            "analysis_info": {
                "total_messages": 0,
                "participants": 0,
                "core_topics": [],
                "main_achievements": ["分析完成，已识别出话题"],
                "pending_items": ["可进一步优化话题分类"]
            },
            "chat_groups": backend_data if isinstance(backend_data, list) else [backend_data]
        }

    def load_data(self):
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

    def call_search_api(self, query: str, search_type: str = "keyword"):
        """调用分工3的搜索API"""
        try:
            # 准备搜索请求数据
            search_request = {
                "query": query,
                "search_type": search_type,
                "top_k": 10
            }

            # 如果有数据文件，也发送给搜索服务
            if st.session_state.data_file:
                search_request["data_file"] = st.session_state.data_file

            # 调用分工3的搜索API
            response = requests.post(
                'http://localhost:8001/api/search',
                json=search_request,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # 转换搜索结果格式
                return self._convert_search_results(result)
            else:
                st.warning(f"搜索服务返回错误: {response.status_code}")
                return []

        except requests.exceptions.ConnectionError:
            st.warning("无法连接到搜索服务，请确保分工3的服务正在运行")
            return []
        except Exception as e:
            st.warning(f"调用搜索服务失败: {str(e)}")
            return []

    def _convert_search_results(self, search_results):
        """搜索API返回的结果转换"""
        converted = []

        # 关键词搜索结果
        if 'keyword_results' in search_results:
            for result in search_results['keyword_results']:
                converted.append({
                    'topic_id': result.get('topic_id', ''),
                    'topic_name': result.get('topic_name', ''),
                    'content': result.get('summaries', [''])[0] if result.get('summaries') else '',
                    'sender': result.get('group_info', {}).get('group_name', ''),
                    'score': result.get('search_score', 0) / 10.0,  # 归一化到0-1
                    'search_type': 'keyword'
                })

        # AI推荐结果
        if 'ai_recommendations' in search_results:
            for result in search_results['ai_recommendations']:
                topic_info = result.get('topic_info', {})
                converted.append({
                    'topic_id': topic_info.get('topic_id', ''),
                    'topic_name': topic_info.get('topic_name', ''),
                    'content': topic_info.get('summaries', [''])[0] if topic_info.get('summaries') else '',
                    'sender': topic_info.get('group_info', {}).get('group_name', ''),
                    'score': result.get('confidence', 0.5),
                    'search_type': 'ai'
                })

        return converted

    def update_topic(self, topic_id: str, new_summary: str):
        """更新话题信息"""
        data = self.load_data()

        # 查找并更新话题
        for group in data.get("chat_groups", []):
            for topic in group.get("topics", []):
                if topic['topic_id'] == topic_id:
                    # 更新摘要
                    if 'summaries' not in topic:
                        topic['summaries'] = []
                    if topic['summaries']:
                        topic['summaries'][0] = new_summary
                    else:
                        topic['summaries'] = [new_summary]
                    break

        # 更新session state中的数据
        st.session_state.analysis_data = data

        # 保存更新后的数据供分工4使用
        updated_file = "output/topics_data_updated.json"
        data_dir = os.path.dirname(updated_file)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        with open(updated_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True

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

        # 群聊选择
        data = self.load_data()
        groups = data.get("chat_groups", [])
        if len(groups) > 1:
            st.sidebar.markdown("### 👥 选择群聊")
            group_options = [f"{group['group_name']} ({len(group.get('topics', []))}个话题)" for group in groups]
            selected_group_index = st.sidebar.selectbox(
                "选择要分析的群聊",
                range(len(groups)),
                format_func=lambda x: group_options[x]
            )
            if selected_group_index is not None:
                st.session_state.current_group = groups[selected_group_index]["group_id"]

        # 筛选选项
        st.sidebar.markdown("### 🔍 筛选选项")
        priority_filter = st.sidebar.multiselect(
            "优先级筛选",
            ["高", "中", "低"],
            default=["高", "中", "低"]
        )

        # 导航
        st.sidebar.markdown("### 🧭 导航")
        page = st.sidebar.radio("选择页面", [
            "📊 分析概览",
            "🗂️ 话题浏览",
            "🕸️ 话题图谱",
            "🔍 智能搜索"
        ])

        # 重置按钮
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 重置所有数据"):
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

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("总消息数", f"{total_messages} 条")
        with col2:
            st.metric("参与人数", f"{len(participants_set)} 人")
        with col3:
            st.metric("总话题数", f"{total_topics} 个")

        st.markdown("---")

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
        """话题浏览页面"""
        st.title("🗂️ 话题浏览")

        if not data.get("chat_groups"):
            st.info("请先上传聊天记录文件并进行分析")
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
            search_term = st.text_input("搜索话题", placeholder="输入关键词搜索...")

        with col2:
            sort_by = st.selectbox("排序方式", ["优先级降序", "相关记录数降序", "名称排序"])

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
            "高": "#FF6B6B",  # 红
            "中": "#4ECDC4",  # 青
            "低": "#45B7D1"  # 蓝
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

                if st.button("编辑", key=f"edit_{topic['topic_id']}"):
                    st.session_state.current_topic = topic['topic_id']
                    st.session_state.edit_mode = True

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
            new_summary = st.text_area("话题摘要", value=current_summary, height=100)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存修改"):
                    if self.update_topic(topic['topic_id'], new_summary):
                        st.success("保存成功！")
                        st.session_state.edit_mode = False
                        st.rerun()
            with col2:
                if st.button("❌ 取消"):
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

            if st.button("返回列表"):
                st.session_state.current_topic = None
                st.rerun()

    def render_topic_graph(self, data):
        """渲染话题关系图谱"""
        st.title("🕸️ 话题关系图谱")

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

    def render_search(self, data):
        """渲染智能搜索页面"""
        st.title("🔍 智能搜索")

        if not data.get("chat_groups"):
            st.info("请先上传聊天记录文件并进行分析")
            return

        # 搜索输入
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_query = st.text_input("输入搜索内容", placeholder="输入关键词或完整句子...")
        with col2:
            search_type = st.selectbox("搜索类型", ["关键词", "语义"])
        with col3:
            st.write("")
            st.write("")
            search_button = st.button("开始搜索", type="primary")

        if search_button and search_query:
            st.write(f"正在搜索: `{search_query}`")

            # 调用分工3的搜索API
            with st.spinner("正在搜索..."):
                search_type_param = "keyword" if search_type == "关键词" else "ai_semantic"
                results = self.call_search_api(search_query, search_type_param)

            if results:
                # 按话题分组显示结果
                results_by_topic = {}
                for result in results:
                    topic_id = result['topic_id']
                    if topic_id not in results_by_topic:
                        # 查找话题详细信息
                        topic_info = None
                        for group in data["chat_groups"]:
                            for topic in group.get("topics", []):
                                if topic['topic_id'] == topic_id:
                                    topic_info = topic
                                    break
                            if topic_info:
                                break

                        if topic_info:
                            results_by_topic[topic_id] = {
                                'topic_name': result['topic_name'],
                                'topic_info': topic_info,
                                'results': [],
                                'max_score': result['score']
                            }
                        else:
                            results_by_topic[topic_id] = {
                                'topic_name': result['topic_name'],
                                'topic_info': None,
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

                st.success(f"找到 {len(results)} 条相关结果，分布在 {len(sorted_topics)} 个话题中")

                for topic_id, topic_data in sorted_topics:
                    with st.expander(
                            f"📌 {topic_data['topic_name']} (相关度: {topic_data['max_score']:.2f}, {len(topic_data['results'])}条结果)"):
                        # 显示话题基本信息
                        if topic_data['topic_info']:
                            if topic_data['topic_info'].get('summaries'):
                                st.write(f"**摘要**: {topic_data['topic_info']['summaries'][0]}")
                            if topic_data['topic_info'].get('priority'):
                                st.write(f"**优先级**: {topic_data['topic_info']['priority']}")

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

    def run(self):
        """运行主应用"""
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


# 运行应用
if __name__ == "__main__":
    frontend = FrontendManager()
    frontend.run()