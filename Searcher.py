import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI


class Searcher:
    def __init__(self, data_file: str = "data.json", token_file: str = "config/api_token.txt"):
        self.data_file = self._resolve_file_path(data_file)
        self.token_file = self._resolve_file_path(token_file)
        self.data = self._load_data()
        self.client = self._init_openai_client()

        # 构建话题层级映射
        self._build_hierarchy_maps()

    def _build_hierarchy_maps(self):
        """构建话题层级映射"""
        self.topic_id_to_parent = {}  # 话题ID -> 父话题ID
        self.topic_id_to_children = {}  # 话题ID -> 子话题ID列表
        self.topic_id_to_type = {}  # 话题ID -> 类型 (major/child/orphan)

        for group in self.data.get('chat_groups', []):
            for topic in group.get('topics', []):
                topic_id = topic['topic_id']

                # 确定话题类型
                if topic.get('is_major', False):
                    self.topic_id_to_type[topic_id] = 'major'
                    self.topic_id_to_children[topic_id] = []
                elif topic.get('parent_id'):
                    self.topic_id_to_type[topic_id] = 'child'
                    parent_id = topic['parent_id']
                    self.topic_id_to_parent[topic_id] = parent_id

                    # 添加到父话题的子话题列表中
                    if parent_id not in self.topic_id_to_children:
                        self.topic_id_to_children[parent_id] = []
                    self.topic_id_to_children[parent_id].append(topic_id)
                else:
                    self.topic_id_to_type[topic_id] = 'orphan'

    def _resolve_file_path(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path

        possible_paths = [
            file_path,
            os.path.join(os.path.dirname(__file__), file_path),
            os.path.join(os.path.dirname(__file__), "..", file_path),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)

        return os.path.abspath(file_path)

    def _load_data(self) -> Dict[str, Any]:
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                content = f.read()

            content = re.sub(r'^\s*#.*$', '', content, flags=re.MULTILINE)

            return json.loads(content)
        except FileNotFoundError:
            raise Exception(f"数据文件 {self.data_file} 未找到。请检查文件路径。")
        except json.JSONDecodeError:
            raise Exception(f"数据文件 {self.data_file} 格式错误")

    def _init_openai_client(self) -> OpenAI:
        try:
            with open(self.token_file, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()

            return OpenAI(
                api_key=api_key,
                base_url="https://api-inference.modelscope.cn/v1/"
            )
        except FileNotFoundError:
            raise Exception(f"API token文件 {self.token_file} 未找到。请检查文件路径。")

    def keyword_search(self, query: str, fields: List[str] = None,
                       group_name: str = None, topic_name: str = None,
                       search_children: bool = True, search_parents: bool = True) -> List[Dict[str, Any]]:
        """
        关键词搜索，支持层级结构

        Args:
            query: 搜索关键词
            fields: 搜索字段
            group_name: 群聊名称过滤
            topic_name: 话题名称过滤
            search_children: 是否搜索子话题
            search_parents: 是否搜索父话题
        """
        if fields is None:
            fields = ['group_name', 'topic_name', 'summaries', 'related_topics']

        results = []
        query_lower = query.lower()
        group_name_lower = group_name.lower() if group_name else None
        topic_name_lower = topic_name.lower() if topic_name else None

        # 用于跟踪已处理的话题ID，避免重复
        processed_ids = set()

        for group in self.data.get('chat_groups', []):
            if group_name_lower and group_name_lower not in group.get('group_name', '').lower():
                continue

            for topic in group.get('topics', []):
                if topic['topic_id'] in processed_ids:
                    continue

                if topic_name_lower and topic_name_lower not in topic.get('topic_name', '').lower():
                    continue

                score = 0
                match_details = []

                if 'group_name' in fields:
                    group_name_val = group.get('group_name', '').lower()
                    if query_lower in group_name_val:
                        score += 3
                        match_details.append(f"群组名称匹配: {group['group_name']}")

                if 'topic_name' in fields:
                    topic_name_val = topic.get('topic_name', '').lower()
                    if query_lower in topic_name_val:
                        score += 3
                        match_details.append(f"主题名称匹配: {topic['topic_name']}")

                if 'summaries' in fields:
                    for i, summary in enumerate(topic.get('summaries', [])):
                        if query_lower in summary.lower():
                            score += 2
                            match_details.append(f"摘要匹配: {summary[:50]}...")

                if 'related_topics' in fields:
                    for related_topic in topic.get('related_topics', []):
                        if query_lower in related_topic.lower():
                            score += 2
                            match_details.append(f"相关主题匹配: {related_topic}")

                if score > 0:
                    # 获取层级信息
                    topic_type = self.topic_id_to_type.get(topic['topic_id'], 'unknown')
                    parent_info = None
                    children_info = []

                    if topic_type == 'child' and search_parents:
                        parent_id = self.topic_id_to_parent.get(topic['topic_id'])
                        if parent_id:
                            parent_topic = self._find_topic_by_id(parent_id, None, None)
                            if parent_topic:
                                parent_info = {
                                    'topic_id': parent_topic['topic_id'],
                                    'topic_name': parent_topic['topic_name']
                                }
                                processed_ids.add(parent_id)

                    elif topic_type == 'major' and search_children:
                        children_ids = self.topic_id_to_children.get(topic['topic_id'], [])
                        for child_id in children_ids[:5]:  # 最多显示5个子话题
                            child_topic = self._find_topic_by_id(child_id, None, None)
                            if child_topic:
                                children_info.append({
                                    'topic_id': child_topic['topic_id'],
                                    'topic_name': child_topic['topic_name']
                                })
                                processed_ids.add(child_id)

                    result = {
                        'topic_id': topic['topic_id'],
                        'topic_name': topic['topic_name'],
                        'priority': topic['priority'],
                        'summaries': topic.get('summaries', []),
                        'related_topics': topic.get('related_topics', []),
                        'group_info': {
                            'group_id': group['group_id'],
                            'group_name': group['group_name'],
                            'description': group['description']
                        },
                        'hierarchy': {
                            'type': topic_type,
                            'parent': parent_info,
                            'children': children_info,
                            'child_count': len(self.topic_id_to_children.get(topic['topic_id'], []))
                        },
                        'search_score': score,
                        'match_details': match_details,
                        'search_type': 'keyword'
                    }
                    results.append(result)
                    processed_ids.add(topic['topic_id'])

        # 按分数排序
        results.sort(key=lambda x: x['search_score'], reverse=True)
        return results

    def ai_semantic_search(self, query: str, max_results: int = 10,
                           group_name: str = None, topic_name: str = None,
                           use_batch_mode: bool = False, batch_size: int = 20,
                           exclude_topic_ids: List[str] = None,
                           consider_hierarchy: bool = True) -> List[Dict[str, Any]]:
        """
        AI语义搜索，支持层级结构

        Args:
            query: 搜索查询
            max_results: 最大结果数
            group_name: 群聊名称过滤
            topic_name: 话题名称过滤
            use_batch_mode: 是否使用批量模式
            batch_size: 批量大小
            exclude_topic_ids: 排除的话题ID
            consider_hierarchy: 是否考虑层级结构
        """
        if use_batch_mode:
            return self._ai_semantic_search_batch(query, max_results, group_name, topic_name, batch_size,
                                                  exclude_topic_ids, consider_hierarchy)
        else:
            return self._ai_semantic_search_single(query, max_results, group_name, topic_name,
                                                   exclude_topic_ids, consider_hierarchy)

    def _ai_semantic_search_single(self, query: str, max_results: int = 10,
                                   group_name: str = None, topic_name: str = None,
                                   exclude_topic_ids: List[str] = None,
                                   consider_hierarchy: bool = True) -> List[Dict[str, Any]]:
        context = self._build_search_context(group_name, topic_name, exclude_topic_ids, consider_hierarchy)

        if not context:
            print("没有找到符合条件的数据用于AI搜索")
            return []

        prompt = f"""
        你是一个微信聊天总结的智能搜索助手。请根据用户查询，从提供的聊天主题数据中找出语义上相关的内容。

        用户查询: "{query}"

        可用的聊天主题数据:
        {context}

        聊天话题采用层级结构:
        1. 大话题 (major): 包含多个相关子话题的主题
        2. 子话题 (child): 属于某个大话题的具体讨论
        3. 独立话题 (orphan): 没有层级关系的话题

        请分析用户查询的意图，并推荐相关的聊天主题。考虑以下因素:
        1. 主题相关性 (即使没有完全匹配的关键词)
        2. 技术概念的关联性
        3. 业务场景的相似性
        4. 用户可能的深层需求
        5. 话题的层级关系（如果查询涉及广泛主题，优先推荐大话题；如果查询具体，推荐子话题）

        重要限制：每个群聊最多只能推荐3个话题！

        请以JSON格式返回结果，包含以下字段:
        - recommended_topics: 推荐的主题ID列表 (最多{max_results}个)
        - reasoning: 推荐理由的简要说明
        - confidence: 整体推荐置信度 (0-1)
        - hierarchy_notes: 关于层级选择的说明（为什么选择大话题或子话题）

        只返回JSON格式的结果，不要其他内容。
        """

        try:
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-Coder-32B-Instruct",
                messages=[
                    {
                        'role': 'system',
                        'content': '你是一个专业的搜索助手，擅长理解技术文档和聊天记录的语义关联。请特别注意话题的层级结构，确保每个群聊最多推荐3个话题。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                temperature=0.3
            )

            ai_response = response.choices[0].message.content
            results = self._parse_ai_response(ai_response, max_results, group_name, topic_name, consider_hierarchy)

            return self._limit_results_per_group(results, max_results)

        except Exception as e:
            print(f"AI搜索出错: {e}")
            return []

    def _ai_semantic_search_batch(self, query: str, max_results: int = 10,
                                  group_name: str = None, topic_name: str = None,
                                  batch_size: int = 20, exclude_topic_ids: List[str] = None,
                                  consider_hierarchy: bool = True) -> List[Dict[str, Any]]:
        all_topics = self._get_all_topics(group_name, topic_name, exclude_topic_ids)

        if not all_topics:
            print("没有找到符合条件的数据用于AI搜索")
            return []

        batches = [all_topics[i:i + batch_size] for i in range(0, len(all_topics), batch_size)]
        all_results = []

        print(f"数据量较大，将分 {len(batches)} 批进行AI搜索...")

        for i, batch in enumerate(batches):
            print(f"正在处理第 {i + 1}/{len(batches)} 批数据...")

            context = self._build_batch_context(batch, consider_hierarchy)

            prompt = f"""
            你是一个微信聊天总结的智能搜索助手。请根据用户查询，从提供的聊天主题数据中找出语义上相关的内容。

            用户查询: "{query}"

            可用的聊天主题数据 (第 {i + 1} 批，共 {len(batches)} 批):
            {context}

            聊天话题采用层级结构:
            1. 大话题 (major): 包含多个相关子话题的主题
            2. 子话题 (child): 属于某个大话题的具体讨论
            3. 独立话题 (orphan): 没有层级关系的话题

            请分析用户查询的意图，并推荐相关的聊天主题。考虑以下因素:
            1. 主题相关性 (即使没有完全匹配的关键词)
            2. 技术概念的关联性
            3. 业务场景的相似性
            4. 用户可能的深层需求
            5. 话题的层级关系（如果查询涉及广泛主题，优先推荐大话题；如果查询具体，推荐子话题）
            6. 不必执着于第5条 ，如果你认为是输入错误或确实无关联，可以返回空
            重要限制：每个群聊最多只能推荐3个话题！

            请以JSON格式返回结果，包含以下字段:
            - recommended_topics: 推荐的主题ID列表
            - reasoning: 推荐理由的简要说明
            - confidence: 整体推荐置信度 (0-1)
            - hierarchy_notes: 关于层级选择的说明

            只返回JSON格式的结果，不要其他内容。
            """

            try:
                response = self.client.chat.completions.create(
                    model="Qwen/Qwen2.5-Coder-32B-Instruct",
                    messages=[
                        {
                            'role': 'system',
                            'content': '你是一个专业的搜索助手，擅长理解技术文档和聊天记录的语义关联。请特别注意话题的层级结构，确保每个群聊最多推荐3个话题。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    temperature=0.3
                )

                ai_response = response.choices[0].message.content
                batch_results = self._parse_ai_response(ai_response, max_results, group_name, topic_name,
                                                        consider_hierarchy)
                all_results.extend(batch_results)

            except Exception as e:
                print(f"第 {i + 1} 批AI搜索出错: {e}")
                continue

        all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return self._limit_results_per_group(all_results, max_results)

    def _build_search_context(self, group_name: str = None, topic_name: str = None,
                              exclude_topic_ids: List[str] = None,
                              consider_hierarchy: bool = True) -> str:
        context_parts = []
        group_name_lower = group_name.lower() if group_name else None
        topic_name_lower = topic_name.lower() if topic_name else None
        exclude_set = set(exclude_topic_ids) if exclude_topic_ids else set()

        for group in self.data.get('chat_groups', []):
            if group_name_lower and group_name_lower not in group.get('group_name', '').lower():
                continue

            for topic in group.get('topics', []):
                if topic_name_lower and topic_name_lower not in topic.get('topic_name', '').lower():
                    continue

                if topic['topic_id'] in exclude_set:
                    continue

                topic_info = self._format_topic_for_context(topic, group, consider_hierarchy)
                context_parts.append(topic_info)

        return "\n".join(context_parts)

    def _build_batch_context(self, batch: List[Dict[str, Any]], consider_hierarchy: bool = True) -> str:
        context_parts = []

        for item in batch:
            topic = item['topic']
            group = item['group']

            topic_info = self._format_topic_for_context(topic, group, consider_hierarchy)
            context_parts.append(topic_info)

        return "\n".join(context_parts)

    def _format_topic_for_context(self, topic: Dict[str, Any], group: Dict[str, Any],
                                  consider_hierarchy: bool = True) -> str:
        """格式化话题信息用于AI上下文"""
        topic_info = [
            f"主题: {topic['topic_name']} (ID: {topic['topic_id']})",
            f"群组: {group['group_name']}",
            f"优先级: {topic['priority']}",
            f"摘要: {'; '.join(topic['summaries'])}"
        ]

        # 添加层级信息
        if consider_hierarchy:
            topic_type = self.topic_id_to_type.get(topic['topic_id'], 'unknown')
            hierarchy_info = f"类型: {topic_type}"

            if topic_type == 'child':
                parent_id = self.topic_id_to_parent.get(topic['topic_id'])
                if parent_id:
                    parent_topic = self._find_topic_by_id(parent_id, None, None)
                    if parent_topic:
                        hierarchy_info += f", 父话题: {parent_topic['topic_name']}"
            elif topic_type == 'major':
                children_count = len(self.topic_id_to_children.get(topic['topic_id'], []))
                hierarchy_info += f", 子话题数量: {children_count}"

            topic_info.append(hierarchy_info)

        if topic.get('related_topics'):
            topic_info.append(f"相关主题: {', '.join(topic['related_topics'])}")

        return "\n".join(topic_info) + "\n---"

    def _get_all_topics(self, group_name: str = None, topic_name: str = None, exclude_topic_ids: List[str] = None) -> \
            List[Dict[str, Any]]:
        all_topics = []
        group_name_lower = group_name.lower() if group_name else None
        topic_name_lower = topic_name.lower() if topic_name else None
        exclude_set = set(exclude_topic_ids) if exclude_topic_ids else set()

        for group in self.data.get('chat_groups', []):
            if group_name_lower and group_name_lower not in group.get('group_name', '').lower():
                continue

            for topic in group.get('topics', []):
                if topic_name_lower and topic_name_lower not in topic.get('topic_name', '').lower():
                    continue

                if topic['topic_id'] in exclude_set:
                    continue

                all_topics.append({
                    'topic': topic,
                    'group': group
                })

        return all_topics

    def _limit_results_per_group(self, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        grouped_results = {}

        for result in results:
            group_id = result['topic_info']['group_info']['group_id']
            if group_id not in grouped_results:
                grouped_results[group_id] = []
            grouped_results[group_id].append(result)

        limited_results = []
        for group_id, group_results in grouped_results.items():
            group_results_sorted = sorted(group_results, key=lambda x: x.get('confidence', 0), reverse=True)
            limited_results.extend(group_results_sorted[:3])

        limited_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return limited_results[:max_results]

    def _parse_ai_response(self, ai_response: str, max_results: int,
                           group_name: str = None, topic_name: str = None,
                           consider_hierarchy: bool = True) -> List[Dict[str, Any]]:
        try:
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())

                recommended_topics = result_data.get('recommended_topics', [])[:max_results * 2]
                results = []

                for topic_id in recommended_topics:
                    topic_info = self._find_topic_by_id(topic_id, group_name, topic_name)
                    if topic_info:
                        # 添加层级信息
                        topic_type = self.topic_id_to_type.get(topic_id, 'unknown')
                        hierarchy_info = {
                            'type': topic_type
                        }

                        if topic_type == 'child':
                            parent_id = self.topic_id_to_parent.get(topic_id)
                            if parent_id:
                                parent_topic = self._find_topic_by_id(parent_id, None, None)
                                if parent_topic:
                                    hierarchy_info['parent'] = {
                                        'topic_id': parent_topic['topic_id'],
                                        'topic_name': parent_topic['topic_name']
                                    }
                        elif topic_type == 'major':
                            children_ids = self.topic_id_to_children.get(topic_id, [])
                            hierarchy_info['child_count'] = len(children_ids)

                        topic_info['hierarchy'] = hierarchy_info

                        results.append({
                            'topic_info': topic_info,
                            'reasoning': result_data.get('reasoning', ''),
                            'confidence': result_data.get('confidence', 0.5),
                            'hierarchy_notes': result_data.get('hierarchy_notes', ''),
                            'search_type': 'ai_semantic'
                        })

                return results
            else:
                print("无法解析AI返回的JSON格式")
                print(f"AI返回内容: {ai_response}")
                return []

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"AI返回内容: {ai_response}")
            return []

    def _find_topic_by_id(self, topic_id: str, group_name: str = None, topic_name: str = None) -> Optional[
        Dict[str, Any]]:
        group_name_lower = group_name.lower() if group_name else None
        topic_name_lower = topic_name.lower() if topic_name else None

        for group in self.data.get('chat_groups', []):
            if group_name_lower and group_name_lower not in group.get('group_name', '').lower():
                continue

            for topic in group.get('topics', []):
                if topic_name_lower and topic_name_lower not in topic.get('topic_name', '').lower():
                    continue

                if topic['topic_id'] == topic_id:
                    result = {
                        'topic_id': topic['topic_id'],
                        'topic_name': topic['topic_name'],
                        'priority': topic['priority'],
                        'summaries': topic['summaries'],
                        'related_topics': topic.get('related_topics', []),
                        'group_info': {
                            'group_id': group['group_id'],
                            'group_name': group['group_name'],
                            'description': group['description']
                        }
                    }
                    return result
        return None

    def search(self, query: str, use_ai: bool = True,
               ai_max_results: int = 10, group_name: str = None, topic_name: str = None,
               use_batch_mode: bool = False, batch_size: int = 20,
               search_children: bool = True, search_parents: bool = True) -> Dict[str, Any]:
        """执行搜索，支持层级结构"""
        keyword_results = self.keyword_search(
            query,
            group_name=group_name,
            topic_name=topic_name,
            search_children=search_children,
            search_parents=search_parents
        )

        ai_results = []
        if use_ai and query.strip():
            exclude_topic_ids = [result['topic_id'] for result in keyword_results]
            ai_results = self.ai_semantic_search(
                query,
                max_results=ai_max_results,
                group_name=group_name,
                topic_name=topic_name,
                use_batch_mode=use_batch_mode,
                batch_size=batch_size,
                exclude_topic_ids=exclude_topic_ids,
                consider_hierarchy=True
            )

        return {
            'query': query,
            'keyword_results': keyword_results,
            'ai_recommendations': ai_results,
            'search_filters': {
                'group_name': group_name,
                'topic_name': topic_name,
                'search_children': search_children,
                'search_parents': search_parents
            },
            'stats': {
                'keyword_matches': len(keyword_results),
                'ai_recommendations': len(ai_results)
            }
        }

    def display_results(self, search_results: Dict[str, Any]):
        print(f"\n搜索查询: {search_results['query']} ")

        filters = search_results['search_filters']
        if filters['group_name'] or filters['topic_name']:
            print("搜索范围:")
            if filters['group_name']:
                print(f"  - 群聊: {filters['group_name']}")
            if filters['topic_name']:
                print(f"  - 话题: {filters['topic_name']}")
            if filters['search_children'] or filters['search_parents']:
                print("  - 层级搜索:")
                if filters['search_parents']:
                    print("    * 搜索父话题")
                if filters['search_children']:
                    print("    * 搜索子话题")
            print()

        print(f"关键词匹配: {search_results['stats']['keyword_matches']} 个")
        print(f"AI推荐: {search_results['stats']['ai_recommendations']} 个\n")

        if search_results['keyword_results']:
            print("关键词匹配结果:")
            for i, result in enumerate(search_results['keyword_results'], 1):
                self._display_single_result(result, i, 'keyword')
                print()

        if search_results['ai_recommendations']:
            print("AI智能推荐 (每个群聊最多3个):")
            for i, result in enumerate(search_results['ai_recommendations'], 1):
                self._display_single_result(result['topic_info'], i, 'ai', result)
                print()

    def _display_single_result(self, result: Dict[str, Any], index: int,
                               search_type: str, ai_info: Dict[str, Any] = None):
        """显示单个搜索结果"""
        # 确定话题类型标签
        type_label = ""
        hierarchy = result.get('hierarchy', {})
        topic_type = hierarchy.get('type', 'unknown')

        if topic_type == 'major':
            type_label = "[大话题]"
        elif topic_type == 'child':
            type_label = "[子话题]"
        elif topic_type == 'orphan':
            type_label = "[独立话题]"

        # 显示基本信息
        if search_type == 'keyword':
            print(f"{index}. {type_label} [{result['group_info']['group_name']}] {result['topic_name']} "
                  f"(优先级: {result['priority']}, 匹配度: {result['search_score']})")
        else:
            confidence = ai_info.get('confidence', 0) if ai_info else 0
            print(f"{index}. {type_label} [{result['group_info']['group_name']}] {result['topic_name']} "
                  f"(置信度: {confidence:.2f})")

        # 显示层级信息
        if topic_type == 'child' and hierarchy.get('parent'):
            parent = hierarchy['parent']
            print(f"   父话题: {parent['topic_name']}")
        elif topic_type == 'major' and hierarchy.get('child_count', 0) > 0:
            print(f"   子话题数量: {hierarchy['child_count']}")
            # 显示部分子话题
            if hierarchy.get('children'):
                children_names = [child['topic_name'] for child in hierarchy['children'][:3]]
                print(f"   部分子话题: {', '.join(children_names)}")
                if hierarchy['child_count'] > 3:
                    print(f"   还有 {hierarchy['child_count'] - 3} 个子话题...")

        # 显示摘要
        if result.get('summaries'):
            print(f"   摘要: {result['summaries'][0][:100]}...")

        # 显示相关话题
        if result.get('related_topics'):
            print(f"   相关主题: {', '.join(result['related_topics'][:3])}")

        # 显示匹配详情（仅关键词搜索）
        if search_type == 'keyword' and result.get('match_details'):
            for detail in result['match_details'][:2]:
                print(f"   - {detail}")

        # 显示AI推荐理由
        if search_type == 'ai' and ai_info and ai_info.get('reasoning'):
            print(f"   推荐理由: {ai_info['reasoning'][:100]}...")

        # 显示层级选择说明
        if search_type == 'ai' and ai_info and ai_info.get('hierarchy_notes'):
            print(f"   层级说明: {ai_info['hierarchy_notes'][:80]}...")

    def get_available_groups(self) -> List[str]:
        groups = []
        for group in self.data.get('chat_groups', []):
            groups.append(group['group_name'])
        return sorted(groups)

    def get_available_topics(self, group_name: str = None) -> List[Dict[str, Any]]:
        topics = []
        group_name_lower = group_name.lower() if group_name else None

        for group in self.data.get('chat_groups', []):
            if group_name_lower and group_name_lower not in group.get('group_name', '').lower():
                continue

            for topic in group.get('topics', []):
                topic_type = self.topic_id_to_type.get(topic['topic_id'], 'unknown')
                topic_info = {
                    'name': topic['topic_name'],
                    'type': topic_type,
                    'priority': topic['priority']
                }

                # 添加层级信息
                if topic_type == 'child':
                    parent_id = self.topic_id_to_parent.get(topic['topic_id'])
                    if parent_id:
                        parent_topic = self._find_topic_by_id(parent_id, None, None)
                        if parent_topic:
                            topic_info['parent'] = parent_topic['topic_name']
                elif topic_type == 'major':
                    topic_info['child_count'] = len(self.topic_id_to_children.get(topic['topic_id'], []))

                topics.append(topic_info)

        return topics

    def get_topic_hierarchy(self, group_name: str = None) -> Dict[str, Any]:
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

        group_name_lower = group_name.lower() if group_name else None

        for group in self.data.get('chat_groups', []):
            if group_name_lower and group_name_lower not in group.get('group_name', '').lower():
                continue

            for topic in group.get('topics', []):
                topic_type = self.topic_id_to_type.get(topic['topic_id'], 'unknown')

                if topic_type == 'major':
                    children_ids = self.topic_id_to_children.get(topic['topic_id'], [])
                    children = []

                    for child_id in children_ids:
                        child_topic = self._find_topic_by_id(child_id, None, None)
                        if child_topic:
                            children.append({
                                'id': child_topic['topic_id'],
                                'name': child_topic['topic_name'],
                                'priority': child_topic['priority']
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

                elif topic_type == 'orphan':
                    hierarchy['orphan_topics'].append({
                        'id': topic['topic_id'],
                        'name': topic['topic_name'],
                        'group_id': group['group_id'],
                        'group_name': group['group_name'],
                        'priority': topic['priority']
                    })
                    hierarchy['statistics']['total_orphan'] += 1

        return hierarchy


def main():
    data_file = input("请输入数据文件路径 (默认: data.json): ").strip()
    if not data_file:
        data_file = "data.json"

    token_file = input("请输入API token文件路径 (默认: api_token.txt): ").strip()
    if not token_file:
        token_file = "api_token.txt"

    try:
        searcher = Searcher(data_file, token_file)
        print(f"成功加载数据文件: {searcher.data_file}")
        print(f"成功加载token文件: {searcher.token_file}")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请检查文件路径是否正确，然后重新运行程序。")
        return

    while True:
        print("\n=== 微信聊天记录搜索系统 (支持层级结构) ===")
        print("可选操作:")
        print("1. 普通搜索")
        print("2. 指定群聊搜索")
        print("3. 指定话题搜索")
        print("4. 查看可用群聊")
        print("5. 查看话题层级结构")
        print("6. 查看可用话题")
        print("7. 退出")

        choice = input("\n请选择操作 (1-7): ").strip()

        if choice == '7':
            break
        elif choice == '4':
            groups = searcher.get_available_groups()
            print("\n可用群聊:")
            for i, group in enumerate(groups, 1):
                print(f"{i}. {group}")
            continue
        elif choice == '5':
            group_name = input("请输入要查看层级结构的群聊名称 (留空查看所有): ").strip() or None
            hierarchy = searcher.get_topic_hierarchy(group_name)

            scope = f"群聊 '{group_name}' 的" if group_name else "所有"
            print(f"\n{scope}话题层级结构:")
            print(f"统计: 大话题 {hierarchy['statistics']['total_major']} 个, "
                  f"子话题 {hierarchy['statistics']['total_children']} 个, "
                  f"独立话题 {hierarchy['statistics']['total_orphan']} 个")

            if hierarchy['major_topics']:
                print("\n大话题列表:")
                for i, topic in enumerate(hierarchy['major_topics'], 1):
                    print(f"{i}. [{topic['group_name']}] {topic['name']} (子话题: {topic['child_count']} 个)")
                    for j, child in enumerate(topic['children'][:3], 1):
                        print(f"   {j}. {child['name']} (优先级: {child['priority']})")
                    if topic['child_count'] > 3:
                        print(f"   还有 {topic['child_count'] - 3} 个子话题...")
                    print()

            if hierarchy['orphan_topics']:
                print("\n独立话题列表:")
                for i, topic in enumerate(hierarchy['orphan_topics'][:10], 1):
                    print(f"{i}. [{topic['group_name']}] {topic['name']} (优先级: {topic['priority']})")
                if len(hierarchy['orphan_topics']) > 10:
                    print(f"... 还有 {len(hierarchy['orphan_topics']) - 10} 个独立话题")
            continue
        elif choice == '6':
            group_name = input("请输入要查看话题的群聊名称 (留空查看所有): ").strip() or None
            topics = searcher.get_available_topics(group_name)
            scope = f"群聊 '{group_name}' 的" if group_name else "所有"
            print(f"\n{scope}可用话题:")

            # 按类型分组显示
            major_topics = [t for t in topics if t['type'] == 'major']
            child_topics = [t for t in topics if t['type'] == 'child']
            orphan_topics = [t for t in topics if t['type'] == 'orphan']

            if major_topics:
                print("\n大话题:")
                for i, topic in enumerate(major_topics[:10], 1):
                    print(
                        f"{i}. {topic['name']} (优先级: {topic['priority']}, 子话题: {topic.get('child_count', 0)} 个)")

            if child_topics:
                print("\n子话题:")
                for i, topic in enumerate(child_topics[:10], 1):
                    parent_info = f", 父话题: {topic['parent']}" if topic.get('parent') else ""
                    print(f"{i}. {topic['name']} (优先级: {topic['priority']}{parent_info})")

            if orphan_topics:
                print("\n独立话题:")
                for i, topic in enumerate(orphan_topics[:10], 1):
                    print(f"{i}. {topic['name']} (优先级: {topic['priority']})")

            total = len(major_topics) + len(child_topics) + len(orphan_topics)
            print(f"\n总计: {total} 个话题 (大话题: {len(major_topics)}, "
                  f"子话题: {len(child_topics)}, 独立话题: {len(orphan_topics)})")
            continue

        query = input("\n请输入搜索关键词: ").strip()
        if not query:
            continue

        group_name = None
        topic_name = None

        if choice == '2':
            groups = searcher.get_available_groups()
            print("\n可用群聊:")
            for i, group in enumerate(groups, 1):
                print(f"{i}. {group}")
            group_choice = input("\n请选择群聊编号或输入群聊名称: ").strip()
            if group_choice.isdigit() and 1 <= int(group_choice) <= len(groups):
                group_name = groups[int(group_choice) - 1]
            else:
                group_name = group_choice

        elif choice == '3':
            topics = searcher.get_available_topics()
            print("\n可用话题 (前20个):")
            topic_list = []
            for i, topic in enumerate(topics[:20], 1):
                type_label = ""
                if topic['type'] == 'major':
                    type_label = "[大]"
                elif topic['type'] == 'child':
                    type_label = "[子]"

                display_name = f"{type_label} {topic['name']}"
                print(f"{i}. {display_name}")
                topic_list.append(topic['name'])

            topic_choice = input("\n请选择话题编号或输入话题名称: ").strip()
            if topic_choice.isdigit() and 1 <= int(topic_choice) <= len(topic_list):
                topic_name = topic_list[int(topic_choice) - 1]
            else:
                topic_name = topic_choice

        # 层级搜索选项
        print("\n层级搜索选项:")
        search_children = input("是否搜索子话题? (y/n, 默认y): ").strip().lower() != 'n'
        search_parents = input("是否搜索父话题? (y/n, 默认y): ").strip().lower() != 'n'

        use_ai = input("\n是否使用AI搜索? (y/n, 默认y): ").strip().lower() != 'n'

        if use_ai:
            try:
                ai_max_results = int(input("请输入AI搜索结果数量 (默认10): ").strip() or "10")
            except ValueError:
                ai_max_results = 10

            all_topics_count = len(searcher._get_all_topics(group_name, topic_name, None))
            use_batch_mode = all_topics_count > 20

            if use_batch_mode:
                print(f"检测到 {all_topics_count} 个话题，将使用分批处理模式")
                try:
                    batch_size = int(input("请输入每批处理的话题数量 (默认20): ").strip() or "20")
                except ValueError:
                    batch_size = 20
            else:
                batch_size = 20
        else:
            ai_max_results = 10
            use_batch_mode = False
            batch_size = 20

        print("\n正在执行关键词搜索...")
        search_results = searcher.search(
            query,
            use_ai=use_ai,
            ai_max_results=ai_max_results,
            group_name=group_name,
            topic_name=topic_name,
            use_batch_mode=use_batch_mode,
            batch_size=batch_size,
            search_children=search_children,
            search_parents=search_parents
        )

        searcher.display_results(search_results)


if __name__ == "__main__":
    main()