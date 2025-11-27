import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import time
import threading


class TopicGraph:
    """话题图结构 - 基于related_topics建立连接"""
    # 类属性声明
    chat_groups: List[Dict[str, Any]]
    graph: Dict[str, List[str]]
    json_file: str
    def __init__(self, json_file: str) -> None:
        ...

    def load_from_json(self, json_file: str) -> None:
        ...

    def _build_graph_from_data(self) -> None:
        ...

    def _find_topic_id_by_name(self, topic_name: str) -> str:
        ...

    def _count_topics(self) -> int:
        ...

    def save_to_json(self) -> None:
        ...

    def add_topic_from_json_file(self) -> bool:
        ...

    def _add_single_topic_from_data(self, group_id: str, topic_data: Dict) -> None:
        ...

    def get_topic_name_by_id(self, topic_id: str) -> str:
        ...

    def get_topic_details(self, topic_id: str) -> Dict:
        ...

    def show_topic_details(self, topic_id: str) -> None:
        ...

    def list_all_topics(self) -> None:
        ...

    def search_topics(self, keyword: str) -> None:
        ...

    def show_graph_structure(self) -> None:
        .


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
            self.load_from_json(json_file)
        else:
            # 如果没有提供文件，使用默认数据
            self.load_default_data()

    def load_default_data(self):
        """加载默认数据"""
        default_data = {
            "chat_groups": [
                {
                    "group_id": "group_001",
                    "group_name": "技术交流群",
                    "description": "程序员技术讨论群",
                    "topics": [
                        {
                            "topic_id": "topic_001_01",
                            "topic_name": "Python异步编程",
                            "priority": "中",
                            "summaries": [
                                "讨论了asyncio的基本原理和使用方法",
                                "对比了async/await与传统多线程的优劣",
                                "分享了常见异步编程的坑和解决方案"
                            ],
                            "related_records": [
                                "2024-01-15 10:30:15 用户A: 大家有使用过asyncio吗？",
                                "2024-01-15 10:35:22 用户B: 我们项目在用，性能提升很明显",
                                "2024-01-15 10:40:18 用户C: 但是调试比较麻烦，有什么好方法？"
                            ],
                            "related_topics": ["异步IO性能调优", "协程与生成器对比"]
                        },
                        {
                            "topic_id": "topic_001_02",
                            "topic_name": "异步IO性能调优",
                            "priority": "中",
                            "summaries": [
                                "讨论了异步IO在高并发场景下的性能优化",
                                "分享了异步上下文管理的技巧",
                                "总结了异步编程中的内存管理"
                            ],
                            "related_records": [
                                "2024-01-16 14:20:33 用户D: 异步IO在高并发下有什么优化技巧？",
                                "2024-01-16 14:25:47 用户E: 连接池管理很重要",
                                "2024-01-16 14:30:12 用户A: 异步上下文管理器可以减少资源泄漏"
                            ],
                            "related_topics": ["异步数据库连接池", "异步任务队列设计", "Python异步编程"]
                        }
                    ]
                },
                {
                    "group_id": "group_002",
                    "group_name": "产品经理交流群",
                    "description": "产品设计与需求讨论",
                    "topics": [
                        {
                            "topic_id": "topic_002_01",
                            "topic_name": "用户需求分析方法",
                            "priority": "中",
                            "summaries": [
                                "分享了用户访谈的技巧和注意事项",
                                "讨论了如何识别伪需求和核心需求",
                                "总结了需求优先级排序的方法论"
                            ],
                            "related_records": [
                                "2024-01-17 09:15:28 用户A: 用户总说想要更多功能，但实际不用",
                                "2024-01-17 09:20:45 用户B: 要用Jobs-to-be-done框架分析",
                                "2024-01-17 09:25:33 用户C: 我们团队用Kano模型效果不错"
                            ],
                            "related_topics": ["用户画像构建方法", "需求验证实验设计"]
                        }
                    ]
                }
            ]
        }

        self.chat_groups = default_data["chat_groups"]
        self._build_graph_from_data()

    def load_from_json(self, json_file: str):
        """从JSON文件加载"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.chat_groups = data.get('chat_groups', [])
            self._build_graph_from_data()
            print(f"加载成功: {len(self.chat_groups)}个群组, {self._count_topics()}个话题")

        except FileNotFoundError:
            print(f"文件不存在，使用默认数据")
            self.load_default_data()

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

    def save_to_json(self, json_file: str = None):
        """保存到JSON文件"""
        if json_file is None:
            json_file = self.json_file

        data = {
            'chat_groups': self.chat_groups
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"保存到 {json_file}")

    def add_topic_simple(self, group_id: str, topic_name: str, priority: str, description: str = "",
                         related_topics: List[str] = None):
        """简化版添加话题（只包含基本信息）"""
        if related_topics is None:
            related_topics = []

        # 查找群组
        group = None
        for g in self.chat_groups:
            if g['group_id'] == group_id:
                group = g
                break

        if not group:
            print(f"群组 {group_id} 不存在")
            return

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

        print(f"添加话题: {topic_name} (ID: {topic_id})")

        if self.json_file:
            self.save_to_json()

    def add_topic_complete(self, group_id: str, topic_id: str, topic_name: str, priority: str,
                           summaries: List[str], related_records: List[str], related_topics: List[str]):
        """完整版添加话题（包含所有信息）"""
        # 检查话题ID是否已存在
        if self._topic_id_exists(topic_id):
            print(f"话题ID {topic_id} 已存在，请使用不同的ID")
            return

        # 查找群组
        group = None
        for g in self.chat_groups:
            if g['group_id'] == group_id:
                group = g
                break

        if not group:
            print(f"群组 {group_id} 不存在")
            return

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

        print(f"完整添加话题: {topic_name} (ID: {topic_id})")

        if self.json_file:
            self.save_to_json()

    def add_chat_records(self, topic_id: str, new_records: List[str]):
        """向现有话题添加新的聊天记录"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            print(f"话题 {topic_id} 不存在")
            return

        # 查找话题在数据结构中的位置
        for group in self.chat_groups:
            for i, topic_item in enumerate(group.get('topics', [])):
                if topic_item['topic_id'] == topic_id:
                    # 添加新的聊天记录
                    group['topics'][i]['related_records'].extend(new_records)
                    print(f"向话题 '{topic['topic_name']}' 添加了 {len(new_records)} 条聊天记录")

                    if self.json_file:
                        self.save_to_json()
                    return

        print(f"未找到话题 {topic_id}")

    def add_summary(self, topic_id: str, new_summary: str):
        """向现有话题添加新的总结"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            print(f"话题 {topic_id} 不存在")
            return

        # 查找话题在数据结构中的位置
        for group in self.chat_groups:
            for i, topic_item in enumerate(group.get('topics', [])):
                if topic_item['topic_id'] == topic_id:
                    # 添加新的总结
                    group['topics'][i]['summaries'].append(new_summary)
                    print(f"向话题 '{topic['topic_name']}' 添加了总结: {new_summary}")

                    if self.json_file:
                        self.save_to_json()
                    return

        print(f"未找到话题 {topic_id}")

    def add_related_topic(self, topic_id: str, related_topic_name: str):
        """向现有话题添加相关话题"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            print(f"话题 {topic_id} 不存在")
            return

        # 查找相关话题的ID
        related_topic_id = self.topic_name_to_id.get(related_topic_name)
        if not related_topic_id:
            print(f"相关话题 '{related_topic_name}' 不存在")
            return

        if related_topic_id == topic_id:
            print("不能将话题与自己关联")
            return

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

                        print(f"将话题 '{topic['topic_name']}' 与 '{related_topic_name}' 关联")

                        if self.json_file:
                            self.save_to_json()
                    else:
                        print(f"话题 '{topic['topic_name']}' 已与 '{related_topic_name}' 关联")
                    return

        print(f"未找到话题 {topic_id}")

    def _topic_id_exists(self, topic_id: str) -> bool:
        """检查话题ID是否已存在"""
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if topic['topic_id'] == topic_id:
                    return True
        return False

    def get_topic_details(self, topic_id: str) -> Dict:
        """获取话题详细信息"""
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if topic['topic_id'] == topic_id:
                    return topic
        return {}

    def show_topic_chat_records(self, topic_id: str):
        """显示话题的聊天记录"""
        topic = self.get_topic_details(topic_id)
        if not topic:
            print(f"话题 {topic_id} 不存在")
            return

        print(f"\n=== 话题聊天记录: {topic['topic_name']} ===")
        print(f"话题ID: {topic_id}")
        print(f"优先级: {topic['priority']}")

        print("\n话题总结:")
        for i, summary in enumerate(topic.get('summaries', []), 1):
            print(f"{i}. {summary}")

        print("\n相关聊天记录:")
        for record in topic.get('related_records', []):
            print(f"- {record}")

        # 显示相关话题
        related_topics = topic.get('related_topics', [])
        if related_topics:
            print(f"\n相关话题: {', '.join(related_topics)}")

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

    def cleanup_old_topics(self, days: int = None):
        """清理过期话题（基于最后访问时间）"""
        if days is None:
            days = self.auto_cleanup_days

        cutoff_time = datetime.now() - timedelta(days=days)
        topics_to_remove = []

        # 这里简化处理：删除长时间未访问的话题
        # 实际应用中可能需要更复杂的逻辑
        print(f"清理 {days} 天前的话题功能待实现")

        if self.json_file:
            self.save_to_json()

    def start_auto_cleanup(self, interval_hours: int = 24):
        """启动自动清理线程"""

        def cleanup_loop():
            while self.running:
                print(f"\n[{datetime.now()}] 执行自动清理...")
                self.cleanup_old_topics()
                time.sleep(interval_hours * 3600)  # 转换为秒

        self.cleanup_thread = threading.Thread(target=cleanup_loop)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
        print(f"自动清理已启动，每{interval_hours}小时执行一次")

    def stop_auto_cleanup(self):
        """停止自动清理线程"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=1)
        print("自动清理已停止")

    def show_graph(self):
        """显示图结构"""
        print("\n=== 话题图结构 ===")

        # 显示按优先级排序的话题
        sorted_topics = self.get_sorted_topics()
        print(f"\n话题列表 (按优先级排序, 共{len(sorted_topics)}个):")
        for i, (topic_id, topic_name, priority) in enumerate(sorted_topics, 1):
            print(f"{i}. {topic_name} (优先级: {priority}, ID: {topic_id})")

        # 显示连接关系
        print("\n话题连接:")
        total_connections = 0
        for topic_id, connections in self.graph.items():
            if connections:
                topic_name = self.topic_id_to_name.get(topic_id, "未知话题")
                connected_topic_names = []
                for connected_id in connections:
                    connected_name = self.topic_id_to_name.get(connected_id, "未知话题")
                    connected_topic_names.append(connected_name)

                if connected_topic_names:
                    print(f"{topic_name} -> {', '.join(connected_topic_names)}")
                    total_connections += len(connections)

        print(f"\n总连接数: {total_connections // 2} (双向连接)")
        print(f"自动清理天数: {self.auto_cleanup_days}天")

    def update_settings(self, auto_cleanup_days: int = None):
        """更新设置"""
        if auto_cleanup_days is not None:
            self.auto_cleanup_days = auto_cleanup_days
            print(f"更新自动清理天数为: {auto_cleanup_days}天")

    def list_all_groups(self):
        """列出所有群组"""
        print("\n=== 所有群组 ===")
        for group in self.chat_groups:
            print(f"\n群组ID: {group['group_id']}")
            print(f"群组名称: {group['group_name']}")
            print(f"描述: {group['description']}")
            print(f"话题数量: {len(group.get('topics', []))}")

    def search_topic(self, keyword: str):
        """搜索话题"""
        print(f"\n=== 搜索话题: '{keyword}' ===")
        found = False
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if keyword.lower() in topic['topic_name'].lower() or \
                        keyword.lower() in ' '.join(topic.get('summaries', [])).lower():
                    print(f"\n找到话题: {topic['topic_name']} (ID: {topic['topic_id']})")
                    print(f"所属群组: {group['group_name']}")
                    print(f"优先级: {topic['priority']}")
                    found = True

        if not found:
            print("未找到相关话题")

    def find_topic_by_id_or_name(self, identifier: str):
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

    def show_topic_by_id_or_name(self, identifier: str):
        """通过ID或名称显示话题详情"""
        topic = self.find_topic_by_id_or_name(identifier)
        if topic:
            self.show_topic_chat_records(topic['topic_id'])
        else:
            print(f"未找到话题: {identifier}")


# 交互式界面
def interactive_interface():
    """交互式界面"""
    print("=== 话题图管理系统（支持添加聊天记录）===")

    # 获取初始设置
    auto_cleanup_days = int(input("设置自动清理天数 (默认30): ") or "30")

    # 创建图
    json_file = input("输入JSON文件路径 (直接回车使用默认数据): ").strip()
    if json_file:
        graph = TopicGraph(json_file, auto_cleanup_days)
    else:
        graph = TopicGraph("chat_topics.json", auto_cleanup_days)

    # 启动自动清理（可选）
    start_auto_cleanup = input("是否启动自动清理? (y/n, 默认n): ").lower() == 'y'
    if start_auto_cleanup:
        interval = int(input("清理间隔(小时, 默认24): ") or "24")
        graph.start_auto_cleanup(interval)

    while True:
        print("\n" + "=" * 60)
        print("1. 显示图结构")
        print("2. 显示所有群组")
        print("3. 简化添加话题")
        print("4. 完整添加话题")
        print("5. 查看话题聊天记录")
        print("6. 添加聊天记录到现有话题")
        print("7. 添加总结到现有话题")
        print("8. 添加相关话题")
        print("9. 搜索话题")
        print("10. 立即清理过期话题")
        print("11. 更新设置")
        print("12. 退出")

        choice = input("请选择操作 (1-12): ").strip()

        if choice == "1":
            graph.show_graph()

        elif choice == "2":
            graph.list_all_groups()

        elif choice == "3":
            graph.list_all_groups()
            group_id = input("输入群组ID: ")
            topic_name = input("输入话题名称: ")
            priority = input("输入优先级 (高/中/低, 默认中): ") or "中"
            description = input("输入话题描述: ")

            # 输入相关话题
            related_topics_input = input("输入相关话题 (用逗号分隔, 直接回车跳过): ")
            related_topics = [t.strip() for t in related_topics_input.split(",")] if related_topics_input else []

            graph.add_topic_simple(group_id, topic_name, priority, description, related_topics)

        elif choice == "4":
            print("\n=== 完整添加话题 ===")
            graph.list_all_groups()

            group_id = input("输入群组ID: ")
            topic_id = input("输入话题ID (如: topic_001_08): ")
