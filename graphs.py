import json
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import defaultdict
import hashlib
import uuid
import time
import threading
from datetime import datetime
import numpy as np
import os

# 导入核心计算库
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI库未安装，API功能将不可用")

from collections import Counter


class TopicGraph:
    """
    话题图管理器 - 修复合并问题版本
    """

    def __init__(self, json_file: str = None,
                 auto_cleanup_days: int = 30,
                 similarity_threshold: float = 0.3,
                 enable_api: bool = False,
                 api_key: str = "",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 debug_mode: bool = True):

        # 确保输出目录存在
        if json_file:
            output_dir = os.path.dirname(json_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                if debug_mode:
                    print(f"📁 创建目录: {output_dir}")

        self.chat_groups: List[Dict[str, Any]] = []
        self.json_file = json_file
        self.similarity_threshold = similarity_threshold
        self.enable_api = enable_api
        self.api_key = api_key
        self.base_url = base_url
        self.auto_cleanup_days = auto_cleanup_days
        self.debug_mode = debug_mode

        # 初始化图结构相关属性
        self.graph = {}
        self.topic_id_to_name = {}
        self.topic_name_to_id = {}
        self.parent_child_map = {}
        self.child_parent_map = {}
        self.topic_id_to_type = {}
        self.topic_embeddings = {}

        # 初始化OpenAI客户端
        self.client = None
        if self.enable_api and OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except Exception as e:
                print(f"⚠️ OpenAI客户端初始化失败: {e}")
                self.client = None

        # 初始化TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._jieba_tokenizer,
            min_df=1,
            max_df=0.8,
            use_idf=True,
            smooth_idf=True
        )
        self._fit_vectorizer_vocab()

        # 自动清理线程相关
        self.running = False
        self.cleanup_thread = None

        # 调试信息存储
        self.debug_logs = []
        self.similarity_calculations = []

        if json_file:
            success = self.load_from_json(json_file)
            if success:
                self._debug_print(f"✅ 从{json_file}加载了{len(self.chat_groups)}个群聊")
                self._debug_current_structure()
            else:
                self._debug_print(f"⚠️ 无法从{json_file}加载数据，将使用空结构")

    def _debug_print(self, message: str, level: str = "INFO"):
        """调试输出"""
        if self.debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] [{level}] {message}"
            print(formatted_message)
            self.debug_logs.append(formatted_message)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """使用考虑词频的Jaccard相似度"""
        if not text1 or not text2:
            return 0.0

        words1 = list(jieba.cut(text1))
        words2 = list(jieba.cut(text2))

        if not words1 or not words2:
            return 0.0

        counter1 = Counter(words1)
        counter2 = Counter(words2)

        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())

        return intersection / union if union > 0 else 0.0

    def add_topic_simple(self, group_id: str, topic_name: str, priority: str,
                         description: str = "", related_topics: List[str] = None) -> Tuple[bool, str]:
        """
        添加话题（自动计算相似度并合并）- 修复合并逻辑

        主要修复点：
        1. 确保找到相似话题后执行合并
        2. 正确处理合并决策
        """
        if related_topics is None:
            related_topics = []

        self._debug_print(f"🚀 开始添加话题: {topic_name}", "TOPIC_ADD")

        # 查找群组
        group = None
        for g in self.chat_groups:
            if g['group_id'] == group_id:
                group = g
                break

        if not group:
            self._debug_print(f"❌ 群组 {group_id} 不存在", "ERROR")
            return False, f"群组 {group_id} 不存在"

        # 检查是否有完全重复的话题
        for topic in group.get('topics', []):
            if topic['topic_name'] == topic_name:
                self._debug_print(f"❌ 话题 '{topic_name}' 已存在", "ERROR")
                return False, f"话题 '{topic_name}' 已存在"

        # 创建新话题对象
        new_topic = {
            "topic_id": f"topic_{group_id.replace('group_', '')}_{len(group['topics']) + 1:04d}",
            "topic_name": topic_name,
            "priority": priority,
            "summaries": [description] if description else [],
            "related_records": [],
            "related_topics": related_topics,
            "is_major": False,
            "parent_id": None
        }

        # 首先添加新话题到群组
        group['topics'].append(new_topic)
        self._debug_print(f"📝 新话题已添加到群组", "TOPIC_ADD")

        # 第一步：查找相似话题并记录
        self._debug_print(f"🔍 开始相似度扫描 (阈值={self.similarity_threshold})", "SIMILARITY_SCAN")

        # 输出现有话题列表（排除新话题自己）
        existing_topics = [t for t in group.get('topics', [])
                           if t['topic_id'] != new_topic['topic_id'] and not t.get('parent_id')]

        self._debug_print(f"  现有独立话题数: {len(existing_topics)}", "SIMILARITY_SCAN")
        for i, t in enumerate(existing_topics, 1):
            self._debug_print(f"    {i}. {t['topic_name']}", "SIMILARITY_SCAN")

        similar_topic_ids = []
        similarity_details = []
        topics_scanned = 0

        for topic in existing_topics:
            topics_scanned += 1
            similarity = self._calculate_topic_similarity(topic, new_topic)

            self._debug_print(
                f"  🎯 与 '{topic['topic_name']}' 的相似度: {similarity:.4f} "
                f"{'✅ 超过阈值' if similarity > self.similarity_threshold else '❌ 未超过'}",
                "SIMILARITY_SCAN")

            if similarity > self.similarity_threshold:
                similar_topic_ids.append({
                    'id': topic['topic_id'],
                    'name': topic['topic_name'],
                    'similarity': similarity
                })
                similarity_details.append({
                    'topic1': new_topic['topic_name'],
                    'topic2': topic['topic_name'],
                    'similarity': similarity
                })

        self._debug_print(f"📊 扫描完成: 检查了{topics_scanned}个话题, 发现{len(similar_topic_ids)}个相似话题",
                          "SIMILARITY_SCAN")

        # 第二步：根据相似话题情况进行处理
        if similar_topic_ids:
            self._debug_print(f"🤝 发现相似话题，开始合并处理...", "MERGE")
            self._debug_print(f"  发现 {len(similar_topic_ids)} 个相似话题:", "MERGE")
            for sim in similar_topic_ids:
                self._debug_print(f"    - {sim['name']} (相似度: {sim['similarity']:.4f})", "MERGE")

            # 收集所有要合并的话题ID
            all_topic_ids = [new_topic['topic_id']]
            all_topic_names = [new_topic['topic_name']]

            for sim_topic in similar_topic_ids:
                all_topic_ids.append(sim_topic['id'])
                all_topic_names.append(sim_topic['name'])

            # 执行合并
            return self._create_major_topic_from_topics(
                group_id=group_id,
                topic_ids=all_topic_ids,
                topic_names=all_topic_names,
                similarity_details=similarity_details
            )
        else:
            # 没有相似话题，作为独立话题
            self._debug_print(f"📌 没有发现相似话题，作为独立话题", "TOPIC_ADD")

            # 重建图结构
            self._build_enhanced_graph()

            if self.json_file:
                self.save_to_json()

            self._debug_print(f"✅ 话题 '{topic_name}' 已作为独立话题添加", "SUCCESS")
            self._debug_current_structure()

            return True, f"话题 '{topic_name}' 已作为独立话题添加"

    def _create_major_topic_from_topics(self, group_id: str, topic_ids: List[str],
                                        topic_names: List[str], similarity_details: List[Dict]) -> Tuple[bool, str]:
        """从多个话题创建大话题（修复版）"""
        if len(topic_ids) < 2:
            self._debug_print(f"❌ 需要至少2个话题才能创建大话题", "ERROR")
            return False, "需要至少2个话题才能创建大话题"

        self._debug_print(f"🏗️ 创建新的大话题，包含{len(topic_ids)}个话题", "MERGE")

        # 查找群组
        group = None
        for g in self.chat_groups:
            if g['group_id'] == group_id:
                group = g
                break

        if not group:
            self._debug_print(f"❌ 群组 {group_id} 不存在", "ERROR")
            return False, f"群组 {group_id} 不存在"

        # 验证所有话题都存在
        valid_topic_ids = []
        valid_topic_names = []

        for topic_id, topic_name in zip(topic_ids, topic_names):
            topic = self.get_topic_details(topic_id)
            if topic:
                # 检查话题状态
                if topic.get('is_major'):
                    self._debug_print(f"  ⚠️ 跳过大话题: {topic_name}", "MERGE")
                    continue

                if topic.get('parent_id'):
                    self._debug_print(f"  ⚠️ 跳过已有父话题的话题: {topic_name}", "MERGE")
                    continue

                valid_topic_ids.append(topic_id)
                valid_topic_names.append(topic_name)
                self._debug_print(f"  ✅ 可合并话题: {topic_name}", "MERGE")
            else:
                self._debug_print(f"  ⚠️ 未找到话题: {topic_id} ({topic_name})", "MERGE")

        if len(valid_topic_ids) < 2:
            self._debug_print(f"❌ 有效可合并话题不足2个", "ERROR")
            return False, "有效可合并话题不足2个"

        # 生成大话题名称
        major_topic_name = self._generate_major_topic_name(valid_topic_names)
        self._debug_print(f"  生成的大话题名称: {major_topic_name}", "MERGE")

        # 创建大话题ID
        major_topic_id = f"major_{group_id.replace('group_', '')}_{uuid.uuid4().hex[:8]}"

        # 创建大话题
        major_topic = {
            "topic_id": major_topic_id,
            "topic_name": major_topic_name,
            "priority": "中",
            "summaries": [f"包含子话题：{', '.join(valid_topic_names)}"],
            "related_records": [],
            "related_topics": [],
            "is_major": True,
            "parent_id": None,
            "child_count": len(valid_topic_ids)
        }

        # 添加大话题到群组
        group['topics'].append(major_topic)
        self._debug_print(f"  大话题已添加到群组", "MERGE")

        # 更新子话题的父ID
        success_count = 0
        for topic_id in valid_topic_ids:
            success = self._update_topic_parent(topic_id, major_topic_id)
            if success:
                success_count += 1
                topic_name = self.get_topic_name_by_id(topic_id)
                self._debug_print(f"  更新子话题父ID: {topic_name} -> {major_topic_name}", "MERGE")
            else:
                self._debug_print(f"  ❌ 更新子话题父ID失败: {topic_id}", "MERGE")

        # 重建图结构
        self._build_enhanced_graph()

        if self.json_file:
            self.save_to_json()

        self._debug_print(f"✅ 成功创建大话题 '{major_topic_name}'，包含 {success_count} 个子话题", "SUCCESS")
        self._debug_current_structure()

        return True, f"✅ 成功创建大话题 '{major_topic_name}'，包含 {success_count} 个相似话题"

    def _update_topic_parent(self, topic_id: str, parent_id: str) -> bool:
        """更新话题的父话题"""
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if topic['topic_id'] == topic_id:
                    topic['parent_id'] = parent_id
                    return True
        return False

    def _calculate_topic_similarity(self, topic1: Dict, topic2: Dict) -> float:
        """计算两个话题的综合相似度"""
        self._debug_print(f"🔍 计算话题相似度:", "SIMILARITY")
        self._debug_print(f"  话题A: {topic1.get('topic_name', '')}", "SIMILARITY")
        self._debug_print(f"  话题B: {topic2.get('topic_name', '')}", "SIMILARITY")

        # 1. 话题名称相似度
        name_sim = self._calculate_text_similarity(
            topic1.get('topic_name', ''),
            topic2.get('topic_name', '')
        )

        # 2. 字符串前缀相似度
        prefix_sim = self._calculate_prefix_similarity(
            topic1.get('topic_name', ''),
            topic2.get('topic_name', '')
        )

        # 3. 摘要相似度
        summary1 = ' '.join(topic1.get('summaries', []))
        summary2 = ' '.join(topic2.get('summaries', []))
        summary_sim = self._calculate_text_similarity(summary1, summary2)

        # 4. 话题相似度和摘要相似度取最大值
        topic_similarity = name_sim
        max_topic_summary_sim = max(topic_similarity, summary_sim)

        # 综合计算
        total_similarity = (max_topic_summary_sim * 0.4) + (prefix_sim * 0.6)

        self._debug_print(f"  名称相似度: {name_sim:.4f}", "SIMILARITY")
        self._debug_print(f"  前缀相似度: {prefix_sim:.4f}", "SIMILARITY")
        self._debug_print(f"  摘要相似度: {summary_sim:.4f}", "SIMILARITY")
        self._debug_print(f"  话题相似度: {topic_similarity:.4f}", "SIMILARITY")
        self._debug_print(f"  max(话题,摘要): {max_topic_summary_sim:.4f}", "SIMILARITY")
        self._debug_print(f"  综合相似度: {total_similarity:.4f}", "SIMILARITY")
        self._debug_print(
            f"  阈值({self.similarity_threshold}): {'✅ 超过' if total_similarity > self.similarity_threshold else '❌ 未超过'}",
            "SIMILARITY")

        # 记录相似度计算
        self.similarity_calculations.append({
            'topic1': topic1.get('topic_name', ''),
            'topic2': topic2.get('topic_name', ''),
            'name_sim': name_sim,
            'prefix_sim': prefix_sim,
            'summary_sim': summary_sim,
            'total_similarity': total_similarity,
            'timestamp': datetime.now().isoformat()
        })

        return total_similarity

    def _calculate_prefix_similarity(self, text1: str, text2: str) -> float:
        """计算字符串前缀相似度"""
        if not text1 or not text2:
            return 0.0
        min_len = min(len(text1), len(text2))
        if min_len == 0:
            return 0.0
        common = 0
        for i in range(min_len):
            if text1[i] == text2[i]:
                common += 1
            else:
                break
        return common / max(len(text1), len(text2))

    def _jieba_tokenizer(self, text: str) -> List[str]:
        """使用jieba进行分词"""
        if not text or not text.strip():
            return []
        words = jieba.lcut(text.strip(), cut_all=False)
        filtered_words = [w for w in words if w.strip() and len(w.strip()) > 1]
        return filtered_words

    def _fit_vectorizer_vocab(self):
        """使用一些初始话题名称拟合TF-IDF向量化器的词汇表"""
        initial_texts = [
            "比赛经历", "比赛奖励", "学习讨论", "工作交流",
            "技术分享", "项目经验", "问题解答", "日常聊天"
        ]
        try:
            self.vectorizer.fit_transform(initial_texts)
        except Exception as e:
            self._debug_print(f"⚠️ TF-IDF向量化器初始化失败: {e}", "WARNING")

    def load_from_json(self, json_file: str) -> bool:
        """从JSON文件加载"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.chat_groups = data.get('chat_groups', [])
            self.json_file = json_file

            # 构建图结构
            self._build_enhanced_graph()

            self._debug_print(f"✅ 成功从 {json_file} 加载 {len(self.chat_groups)} 个群聊", "INFO")
            return True
        except FileNotFoundError:
            self._debug_print(f"⚠️ 文件未找到: {json_file}", "WARNING")
            return False
        except json.JSONDecodeError as e:
            self._debug_print(f"⚠️ JSON解析错误: {e}", "ERROR")
            return False
        except Exception as e:
            self._debug_print(f"⚠️ 加载文件时出错: {e}", "ERROR")
            return False

    def get_topic_details(self, topic_id: str) -> Optional[Dict]:
        """获取话题详细信息"""
        if not topic_id:
            return None

        for group in self.chat_groups:
            for topic in group.get('topics', []):
                if topic['topic_id'] == topic_id:
                    return topic
        return None

    def get_topic_name_by_id(self, topic_id: str) -> str:
        """根据话题ID获取话题名称"""
        if not topic_id:
            return ""

        topic = self.get_topic_details(topic_id)
        return topic['topic_name'] if topic else ""

    def save_to_json(self, json_file: str = None) -> bool:
        """保存到JSON文件"""
        if json_file is None:
            json_file = self.json_file

        if not json_file:
            self._debug_print("⚠️ 未指定保存文件路径", "WARNING")
            return False

        # 确保目录存在
        output_dir = os.path.dirname(json_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data = {
            'chat_groups': self.chat_groups,
            'metadata': {
                'similarity_threshold': self.similarity_threshold,
                'enable_api': self.enable_api,
                'generated_at': datetime.now().isoformat(),
                'debug_logs': self.debug_logs[-100:] if self.debug_mode else [],
                'similarity_calculations': self.similarity_calculations[-50:] if self.debug_mode else []
            }
        }

        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._debug_print(f"✅ 数据已保存到 {json_file}", "INFO")
            return True
        except Exception as e:
            self._debug_print(f"❌ 保存失败: {e}", "ERROR")
            return False

    def _generate_major_topic_name(self, subtopic_names: List[str]) -> str:
        """生成大话题名称"""
        if not subtopic_names:
            return "综合讨论"

        if not self.enable_api or not self.client:
            if len(subtopic_names) == 1:
                return f"关于{subtopic_names[0]}的讨论"
            else:
                return f"综合讨论：{subtopic_names[0]}等"

        try:
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {
                        'role': 'system',
                        'content': '你是一个专业的聊天话题分析助手。请根据给定的子话题，生成一个简洁、准确的大话题名称。'
                    },
                    {
                        'role': 'user',
                        'content': f"请为以下子话题生成一个合适的大话题名称：{', '.join(subtopic_names)}。要求：1. 简洁明了 2. 涵盖所有子话题 3. 不超过15个字"
                    }
                ],
                max_tokens=50,
                temperature=0.7
            )

            major_topic_name = response.choices[0].message.content.strip()
            major_topic_name = major_topic_name.strip('"\'')
            return major_topic_name
        except Exception as e:
            self._debug_print(f"⚠️ API调用失败: {e}", "WARNING")
            return f"综合话题：{subtopic_names[0]}等"

    def get_topic_hierarchy(self, group_id: str = None) -> Dict:
        """获取话题层级结构"""
        hierarchy = {
            'major_topics': [],
            'orphan_topics': [],
            'statistics': {
                'total_major': 0,
                'total_children': 0,
                'total_orphan': 0
            }
        }

        for group in self.chat_groups:
            if group_id and group['group_id'] != group_id:
                continue

            for topic in group.get('topics', []):
                if topic.get('is_major'):
                    children = []
                    child_ids = self.parent_child_map.get(topic['topic_id'], [])
                    for child_id in child_ids:
                        child = self.get_topic_details(child_id)
                        if child:
                            children.append({
                                'id': child['topic_id'],
                                'name': child['topic_name'],
                                'priority': child['priority']
                            })

                    hierarchy['major_topics'].append({
                        'id': topic['topic_id'],
                        'name': topic['topic_name'],
                        'group_id': group['group_id'],
                        'group_name': group['group_name'],
                        'children': children,
                        'child_count': len(children)
                    })
                    hierarchy['statistics']['total_major'] += 1
                    hierarchy['statistics']['total_children'] += len(children)
                elif not topic.get('parent_id'):
                    hierarchy['orphan_topics'].append({
                        'id': topic['topic_id'],
                        'name': topic['topic_name'],
                        'group_id': group['group_id'],
                        'group_name': group['group_name'],
                        'priority': topic['priority']
                    })
                    hierarchy['statistics']['total_orphan'] += 1

        return hierarchy

    def _build_enhanced_graph(self):
        """构建增强的图结构"""
        self.graph.clear()
        self.topic_id_to_name.clear()
        self.topic_name_to_id.clear()
        self.parent_child_map.clear()
        self.child_parent_map.clear()
        self.topic_id_to_type.clear()
        self.topic_embeddings.clear()

        # 第一遍：收集所有话题
        for group in self.chat_groups:
            group_id = group['group_id']
            for topic in group.get('topics', []):
                topic_id = topic['topic_id']
                topic_name = topic['topic_name']

                self.topic_id_to_name[topic_id] = topic_name
                self.topic_name_to_id[topic_name] = topic_id

                self.graph[topic_id] = {
                    'id': topic_id,
                    'name': topic_name,
                    'group_id': group_id,
                    'is_major': topic.get('is_major', False),
                    'parent_id': topic.get('parent_id'),
                    'children': [],
                    'related': []
                }

        # 第二遍：构建类型映射和关系
        for group in self.chat_groups:
            for topic in group.get('topics', []):
                topic_id = topic['topic_id']

                if topic.get('is_major'):
                    self.topic_id_to_type[topic_id] = 'major'
                elif topic.get('parent_id'):
                    self.topic_id_to_type[topic_id] = 'child'
                else:
                    self.topic_id_to_type[topic_id] = 'orphan'

                parent_id = topic.get('parent_id')
                if parent_id and parent_id in self.graph:
                    self.child_parent_map[topic_id] = parent_id
                    if parent_id not in self.parent_child_map:
                        self.parent_child_map[parent_id] = []
                    self.parent_child_map[parent_id].append(topic_id)

                    if parent_id in self.graph:
                        self.graph[parent_id]['children'].append(topic_id)

                for related_topic_name in topic.get('related_topics', []):
                    related_topic_id = self.topic_name_to_id.get(related_topic_name)
                    if related_topic_id and related_topic_id != topic_id:
                        if related_topic_id not in self.graph[topic_id]['related']:
                            self.graph[topic_id]['related'].append(related_topic_id)

        self._debug_print(
            f"📊 图结构构建完成: {len(self.graph)}个节点, {sum(len(v['children']) for v in self.graph.values())}条父子关系",
            "GRAPH")

    def _debug_current_structure(self):
        """输出当前话题结构"""
        hierarchy = self.get_topic_hierarchy()

        self._debug_print("📊 当前话题结构:", "STRUCTURE")
        self._debug_print(f"  大话题数: {len(hierarchy.get('major_topics', []))}", "STRUCTURE")
        self._debug_print(f"  子话题数: {hierarchy.get('statistics', {}).get('total_children', 0)}", "STRUCTURE")
        self._debug_print(f"  独立话题: {hierarchy.get('statistics', {}).get('total_orphan', 0)}", "STRUCTURE")

        for i, major in enumerate(hierarchy.get('major_topics', []), 1):
            self._debug_print(f"  {i}. 🏢 {major['name']} ({major['child_count']}个子话题)", "STRUCTURE")
            for j, child in enumerate(major.get('children', []), 1):
                self._debug_print(f"     {j}. 🔗 {child['name']}", "STRUCTURE")

        for i, orphan in enumerate(hierarchy.get('orphan_topics', []), 1):
            self._debug_print(f"  {i}. 🔸 {orphan['name']} (独立话题)", "STRUCTURE")

    def test_similarity_and_merge(self, test_data: List[Tuple[str, str, str]] = None):
        """
        测试相似度计算和合并功能

        关键修复：确保测试数据能触发合并
        """
        if test_data is None:
            test_data = [
                ("支教项目招募", "招募支教项目的志愿者", "高"),
                ("非遗项目招募", "招募非遗项目的参与者", "高"),
                ("2025年挑战杯团队招募非遗多模态检索系统成员",
                 "2025年挑战杯项目，招募非遗多模态检索系统开发成员", "高"),
                ("2025年挑战杯团队招募网页设计和财务管理人员",
                 "2025年挑战杯项目，招募网页设计师和财务管理人员", "高"),
            ]

        self._debug_print("🧪 开始相似度计算和合并测试", "TEST")
        self._debug_print("=" * 60, "TEST")

        for i, (topic_name, description, priority) in enumerate(test_data, 1):
            self._debug_print(f"测试 {i}/{len(test_data)}: 添加话题 '{topic_name}'", "TEST")

            success, message = self.add_topic_simple(
                group_id="group_test_001",
                topic_name=topic_name,
                priority=priority,
                description=description
            )

            if success:
                self._debug_print(f"✅ {message}", "TEST")
            else:
                self._debug_print(f"❌ {message}", "TEST")

            self._debug_print("-" * 40, "TEST")

        self._debug_print("=" * 60, "TEST")
        self._debug_print("🧪 测试完成", "TEST")

        # 显示最终结构
        hierarchy = self.get_topic_hierarchy()
        self._debug_print(f"最终结构统计:", "TEST")
        self._debug_print(f"  大话题数: {len(hierarchy.get('major_topics', []))}", "TEST")
        self._debug_print(f"  子话题数: {hierarchy.get('statistics', {}).get('total_children', 0)}", "TEST")
        self._debug_print(f"  独立话题: {hierarchy.get('statistics', {}).get('total_orphan', 0)}", "TEST")


# 测试代码
if __name__ == "__main__":
    print("🧪 测试TopicGraph合并功能...")

    # 确保output目录存在
    os.makedirs("output", exist_ok=True)

    # 创建测试数据文件
    test_data = {
        "chat_groups": [
            {
                "group_id": "group_test_001",
                "group_name": "测试群聊",
                "description": "用于测试的话题群组",
                "topics": []
            }
        ]
    }

    json_file_path = "output/topic_graph_data.json"

    with open(json_file_path, "w", encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    print(f"✅ 创建测试文件: {json_file_path}")

    # 创建并测试TopicGraph
    topic_graph = TopicGraph(
        json_file=json_file_path,
        similarity_threshold=0.1,  # 使用较低的阈值确保合并
        debug_mode=True
    )

    # 运行测试
    topic_graph.test_similarity_and_merge()

    print("\n" + "=" * 60)
    print("✅ 测试完成！查看上面的输出了解合并过程。")
    print("=" * 60)