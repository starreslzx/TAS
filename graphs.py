import json
from typing import Dict, List, Tuple, Any, Optional
import time
import threading


class TopicGraph:
    """增强版话题图结构 - 支持向现有话题添加聊天记录"""

    def __init__(self, json_file: str = None, auto_cleanup_days: int = 30):
        self.chat_groups: List[Dict[str, Any]] = []  # 聊天群组存储
        self.graph: Dict[str, List[str]] = {}  # 图结构存储（简化版，只存储连接关系）
        self.topic_id_to_name: Dict[str, str] = {}  # 话题ID到名称的映射
        self.topic_name_to_id: Dict[str, str] = {}  # 话题名称到ID的映射
        self.json_file = json_file
        self.auto_cleanup_days = auto_cleanup_days  # 自动清理天数
        self.cleanup_thread = None  # 定期清理线程
        self.running = True  # 控制清理线程运行

        if json_file:
            success = self.load_from_json(json_file)
            if not success:
                # 如果加载失败，初始化为空
                self.chat_groups = []
                self._build_graph_from_data()
        else:
            # 如果没有提供文件，初始化为空
            self.chat_groups = []
            self._build_graph_from_data()

    def load_from_json(self, json_file: str) -> bool:
        """从JSON文件加载，返回是否成功"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.chat_groups = data.get('chat_groups', [])
            self._build_graph_from_data()
            return True
        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False

    def _build_graph_from_data(self):
        """从聊天数据构建图结构"""
        self.graph = {}
        self.topic_id_to_name = {}
        self.topic_name_to_id = {}

        # 收集所有话题
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                topic_id = topic['topic_id']
                topic_name = topic['topic_name']

                # 存储话题映射
                self.topic_id_to_name[topic_id] = topic_name
                self.topic_name_to_id[topic_name] = topic_id

                # 初始化图结构
                self.graph[topic_id] = []

        # 构建话题之间的连接（基于related_topics）
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                topic_id = topic['topic_id']

                # 连接相关话题
                for related_topic_name in topic.get('related_topics', []):
                    # 查找相关话题的ID
                    related_topic_id = self.topic_name_to_id.get(related_topic_name)
                    if related_topic_id and related_topic_id != topic_id:
                        # 添加双向连接
                        if related_topic_id not in self.graph[topic_id]:
                            self.graph[topic_id].append(related_topic_id)

                        if topic_id not in self.graph[related_topic_id]:
                            self.graph[related_topic_id].append(topic_id)

    def _count_topics(self):
        """计算话题总数"""
        count = 0
        for group in self.chat_groups:
            count += len(group.get('topics', []))
        return count

    def _priority_order(self, priority: str) -> int:
        """返回优先级排序顺序"""
        priority_order = {"高": 3, "中": 2, "低": 1}
        return priority_order.get(priority, 2)

    def save_to_json(self, json_file: str = None) -> bool:
        """保存到JSON文件，返回是否成功"""
        if json_file is None:
            json_file = self.json_file

        data = {
            'chat_groups': self.chat_groups
        }

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def add_topic_simple(self, group_id: str, topic_name: str, priority: str, description: str = "",
                         related_topics: List[str] = None) -> Tuple[bool, str]:
        """简化版添加话题，返回(是否成功, 结果消息)"""
        if related_topics is None:
            related_topics = []

        # 查找群组
        group = None
        for g in self.chat_groups:
            if g['group_id'] == group_id:
                group = g
                break

        if not group:
            return False, f"群组 {group_id} 不存在"

        # 创建新话题
        topic_id = f"topic_{group_id}_{len(group['topics']) + 1:02d}"
        new_topic = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "priority": priority,
            "summaries": [description] if description else [],
            "related_records": [],
            "related_topics": related_topics
        }

        group['topics'].append(new_topic)

        # 更新映射
        self.topic_id_to_name[topic_id] = topic_name
        self.topic_name_to_id[topic_name] = topic_id

        # 初始化图结构
        self.graph[topic_id] = []

        # 连接相关话题
        for related_topic_name in related_topics:
            related_topic_id = self.topic_name_to_id.get(related_topic_name)
            if related_topic_id and related_topic_id != topic_id:
                # 添加双向连接
                if related_topic_id not in self.graph[topic_id]:
                    self.graph[topic_id].append(related_topic_id)

                if topic_id not in self.graph[related_topic_id]:
                    self.graph[related_topic_id].append(topic_id)

        if self.json_file:
            self.save_to_json()

        return True, topic_id

    def add_topic_complete(self, group_id: str, topic_id: str, topic_name: str, priority: str,
                           summaries: List[str], related_records: List[str], related_topics: List[str]) -> Tuple[bool, str]:
        """完整版添加话题，返回(是否成功, 结果消息)"""
        # 检查话题ID是否已存在
        if self._topic_id_exists(topic_id):
            return False, f"话题ID {topic_id} 已存在，请使用不同的ID"

        # 查找群组
        group = None
        for g in self.chat_groups:
            if g['group_id'] == group_id:
                group = g
                break

        if not group:
            return False, f"群组 {group_id} 不存在"

        # 创建新话题
        new_topic = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "priority": priority,
            "summaries": summaries,
            "related_records": related_records,
            "related_topics": related_topics
        }

        group['topics'].append(new_topic)

        # 更新映射
        self.topic_id_to_name[topic_id] = topic_name
        self.topic_name_to_id[topic_name] = topic_id

        # 初始化图结构
        self.graph[topic_id] = []

        # 连接相关话题
        for related_topic_name in related_topics:
            related_topic_id = self.topic_name_to_id.get(related_topic_name)
            if related_topic_id and related_topic_id != topic_id:
                # 添加双向连接
                if related_topic_id not in self.graph[topic_id]:
                    self.graph[topic_id].append(related_topic_id)

                if topic_id not in self.graph[related_topic_id]:
                    self.graph[related_topic_id].append(topic_id)

        if self.json_file:
            self.save_to_json()

        return True, topic_id

    def add_chat_records(self, topic_id: str, new_records: List[str]) -> Tuple[bool, str]:
        """向现有话题添加新的聊天记录，返回(是否成功, 结果消息)"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            return False, f"话题 {topic_id} 不存在"

        # 查找话题在数据结构中的位置
        for group in self.chat_groups:
            for i, topic_item in enumerate(group.get('topics', [])):
                if topic_item['topic_id'] == topic_id:
                    # 添加新的聊天记录
                    group['topics'][i]['related_records'].extend(new_records)

                    if self.json_file:
                        self.save_to_json()
                    return True, f"成功添加 {len(new_records)} 条聊天记录到话题 '{topic['topic_name']}'"

        return False, f"未找到话题 {topic_id}"

    def add_summary(self, topic_id: str, new_summary: str) -> Tuple[bool, str]:
        """向现有话题添加新的总结，返回(是否成功, 结果消息)"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            return False, f"话题 {topic_id} 不存在"

        # 查找话题在数据结构中的位置
        for group in self.chat_groups:
            for i, topic_item in enumerate(group.get('topics', [])):
                if topic_item['topic_id'] == topic_id:
                    # 添加新的总结
                    group['topics'][i]['summaries'].append(new_summary)

                    if self.json_file:
                        self.save_to_json()
                    return True, f"成功添加总结到话题 '{topic['topic_name']}'"

        return False, f"未找到话题 {topic_id}"

    def add_related_topic(self, topic_id: str, related_topic_name: str) -> Tuple[bool, str]:
        """向现有话题添加相关话题，返回(是否成功, 结果消息)"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            return False, f"话题 {topic_id} 不存在"

        # 查找相关话题的ID
        related_topic_id = self.topic_name_to_id.get(related_topic_name)
        if not related_topic_id:
            return False, f"相关话题 '{related_topic_name}' 不存在"

        if related_topic_id == topic_id:
            return False, "不能将话题与自己关联"

        # 查找话题在数据结构中的位置
        for group in self.chat_groups:
            for i, topic_item in enumerate(group.get('topics', [])):
                if topic_item['topic_id'] == topic_id:
                    # 添加相关话题
                    if related_topic_name not in group['topics'][i]['related_topics']:
                        group['topics'][i]['related_topics'].append(related_topic_name)

                        # 更新图结构
                        if related_topic_id not in self.graph[topic_id]:
                            self.graph[topic_id].append(related_topic_id)

                        if topic_id not in self.graph[related_topic_id]:
                            self.graph[related_topic_id].append(topic_id)

                        if self.json_file:
                            self.save_to_json()
                        return True, f"成功将话题 '{topic['topic_name']}' 与 '{related_topic_name}' 关联"
                    else:
                        return False, f"话题 '{topic['topic_name']}' 已与 '{related_topic_name}' 关联"

        return False, f"未找到话题 {topic_id}"

    def _topic_id_exists(self, topic_id: str) -> bool:
        """检查话题ID是否已存在"""
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if topic['topic_id'] == topic_id:
                    return True
        return False

    def get_topic_details(self, topic_id: str) -> Optional[Dict]:
        """获取话题详细信息"""
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if topic['topic_id'] == topic_id:
                    return topic
        return None

    def get_topic_chat_records(self, topic_id: str) -> Tuple[bool, Dict]:
        """获取话题的聊天记录，返回(是否成功, 数据字典)"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            return False, {}

        result = {
            'topic_id': topic_id,
            'topic_name': topic['topic_name'],
            'priority': topic['priority'],
            'summaries': topic.get('summaries', []),
            'related_records': topic.get('related_records', []),
            'related_topics': topic.get('related_topics', [])
        }
        return True, result

    def get_sorted_topics(self) -> List[Tuple[str, str, str]]:
        """按优先级排序的话题列表"""
        # 收集所有话题
        all_topics = []
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                all_topics.append((topic['topic_id'], topic['topic_name'], topic['priority']))

        # 按优先级降序排序（高 > 中 > 低）
        sorted_topics = sorted(
            all_topics,
            key=lambda x: self._priority_order(x[2]),  # 优先级顺序
            reverse=True
        )
        return sorted_topics

    def get_graph_structure(self) -> Dict:
        """获取图结构数据"""
        sorted_topics = self.get_sorted_topics()
        topic_list = []
        for i, (topic_id, topic_name, priority) in enumerate(sorted_topics, 1):
            topic_list.append({
                'index': i,
                'topic_id': topic_id,
                'topic_name': topic_name,
                'priority': priority
            })

        connections = []
        total_connections = 0
        for topic_id, connected_ids in self.graph.items():
            if connected_ids:
                topic_name = self.topic_id_to_name.get(topic_id, "未知话题")
                connected_topic_names = []
                for connected_id in connected_ids:
                    connected_name = self.topic_id_to_name.get(connected_id, "未知话题")
                    connected_topic_names.append(connected_name)

                if connected_topic_names:
                    connections.append({
                        'source': topic_name,
                        'targets': connected_topic_names
                    })
                    total_connections += len(connected_ids)

        return {
            'topic_count': len(sorted_topics),
            'topics': topic_list,
            'connections': connections,
            'total_connections': total_connections // 2,
            'auto_cleanup_days': self.auto_cleanup_days
        }

    def list_all_groups(self) -> List[Dict]:
        """列出所有群组"""
        groups = []
        for group in self.chat_groups:
            groups.append({
                'group_id': group['group_id'],
                'group_name': group['group_name'],
                'description': group['description'],
                'topic_count': len(group.get('topics', []))
            })
        return groups

    def search_topic(self, keyword: str) -> List[Dict]:
        """搜索话题"""
        results = []
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if (keyword.lower() in topic['topic_name'].lower() or
                    keyword.lower() in ' '.join(topic.get('summaries', [])).lower()):
                    results.append({
                        'topic_id': topic['topic_id'],
                        'topic_name': topic['topic_name'],
                        'group_name': group['group_name'],
                        'priority': topic['priority'],
                        'summaries': topic.get('summaries', [])
                    })
        return results

    def find_topic_by_id_or_name(self, identifier: str) -> Optional[Dict]:
        """通过ID或名称查找话题"""
        # 先尝试按ID查找
        topic_details = self.get_topic_details(identifier)
        if topic_details:
            return topic_details

        # 再尝试按名称查找
        topic_id = self.topic_name_to_id.get(identifier)
        if topic_id:
            return self.get_topic_details(topic_id)

        return None

    def cleanup_old_topics(self, days: int = None):
        """清理过期话题（基于最后访问时间）"""
        if days is None:
            days = self.auto_cleanup_days

        # 这里简化处理
        if self.json_file:
            self.save_to_json()

    def start_auto_cleanup(self, interval_hours: int = 24):
        """启动自动清理线程"""

        def cleanup_loop():
            while self.running:
                self.cleanup_old_topics()
                time.sleep(interval_hours * 3600)  # 转换为秒

        self.cleanup_thread = threading.Thread(target=cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()

    def stop_auto_cleanup(self):
        """停止自动清理线程"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1)

    def update_settings(self, auto_cleanup_days: int = None):
        """更新设置"""
        if auto_cleanup_days is not None:
            self.auto_cleanup_days = auto_cleanup_days