import json
import os
import re
from datetime import datetime
from typing import List, Dict, Optional, Set

# 文件解析相关库
import pdfplumber
from docx import Document
import docx2txt

# API调用
from openai import OpenAI


class ChatAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化聊天记录分析器
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "Qwen/Qwen2.5-7B-Instruct"
        self.chat_structure = {
            "chat_groups": []
        }
        self.current_records = []  # 保存当前处理的聊天记录

    def parse_file(self, file_path: str) -> List[str]:
        """
        解析上传的文件，支持PDF、DOC、DOCX格式
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)
        elif ext == '.doc':
            return self._parse_doc(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，请上传PDF、DOC或DOCX文件")

    def _parse_pdf(self, file_path: str) -> List[str]:
        """解析PDF文件"""
        records = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # 按行分割并清理空白行
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        records.extend(lines)
        except Exception as e:
            raise Exception(f"PDF解析失败: {str(e)}")

        return self._clean_and_limit_records(records)

    def _parse_docx(self, file_path: str) -> List[str]:
        """解析DOCX文件"""
        try:
            # 方法1: 使用python-docx
            doc = Document(file_path)
            records = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    records.append(paragraph.text.strip())

            # 如果没有内容，尝试使用docx2txt
            if not records:
                text = docx2txt.process(file_path)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                records = lines

        except Exception as e:
            raise Exception(f"DOCX解析失败: {str(e)}")

        return self._clean_and_limit_records(records)

    def _parse_doc(self, file_path: str) -> List[str]:
        """解析DOC文件（旧版Word文档）"""
        try:
            # 对于.doc文件，我们可以先尝试用docx2txt
            import docx2txt
            text = docx2txt.process(file_path)
            records = [line.strip() for line in text.split('\n') if line.strip()]
        except Exception as e:
            # 如果上述方法失败，尝试使用antiword
            # 这里我们提供一个备选方案
            try:
                import subprocess
                # 尝试使用antiword
                result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
                if result.returncode == 0:
                    text = result.stdout
                    records = [line.strip() for line in text.split('\n') if line.strip()]
                else:
                    raise Exception("无法解析.doc文件，请确保安装了antiword或转换为PDF/DOCX格式")
            except Exception as e2:
                raise Exception(f"DOC解析失败: {str(e)}，备选方案也失败: {str(e2)}")

        return self._clean_and_limit_records(records)

    def _clean_and_limit_records(self, records: List[str]) -> List[str]:
        """
        清理记录并限制为100条
        """
        # 清理空白和无效记录
        cleaned = []
        for record in records:
            if record and len(record.strip()) > 3:
                cleaned.append(record.strip())
        # 限制为100条
        return cleaned[:100]

    def analyze_topics(self, group_name: str, chat_records: List[str],
                       existing_structure: Optional[Dict] = None,
                       description: str = "") -> Dict:
        """
        分析聊天记录并生成话题结构
        """
        if not chat_records:
            raise ValueError("聊天记录不能为空")

        # 保存当前处理的聊天记录，用于后续根据区间提取消息
        self.current_records = chat_records

        # 准备现有的群聊话题信息
        existing_topics_info = ""
        if existing_structure and "chat_groups" in existing_structure:
            for group in existing_structure["chat_groups"]:
                if group["group_name"] == group_name:
                    existing_topics_info = self._format_existing_topics(group["topics"])
                    break

        # 准备API调用
        prompt = f"""你是一个专业的聊天记录分析师。请分析以下聊天记录，并识别出不同的讨论话题。

聊天记录总共有 {len(chat_records)} 条，每条记录以换行符分隔。
每条记录前面有索引编号（从1开始），格式为"[索引] 记录内容"

已有的群聊话题：
{existing_topics_info if existing_topics_info else "暂无已有话题"}

需要分析的聊天记录：
{self._format_chat_records_with_indices(chat_records)}

请完成以下任务：
1. 识别聊天记录中的主要话题
2. 为每个话题生成：
   - 话题名称（简洁明了）
   - 优先级（高/中/低）
   - 简要总结（3-5句话）
   - 相关记录的索引区间（格式为[[开始索引, 结束索引], ...]，索引从1开始）
   - 相关话题（与其他话题的关联）

重要要求：
1. 确保所有聊天记录都被分配到至少一个话题中
2. 尽量使话题之间的记录区间不重叠，但如果有必要，可以允许少量重叠
3. 使用连续的区间来表示相关记录
4. 区间可以是多个不连续的部分，如[[1,4],[7,12]]
5. 如果一个记录与多个话题相关，可以分配给多个话题
6. 确保覆盖所有记录的区间，没有遗漏任何记录

请以严格的JSON格式返回，格式如下：
{{
  "topics": [
    {{
      "topic_name": "话题名称",
      "priority": "高/中/低",
      "summaries": ["总结1", "总结2", "总结3"],
      "record_intervals": [[1,4], [7,12]],
      "related_topics": ["相关话题1", "相关话题2"]
    }}
  ]
}}

注意：
1. 话题名称要简洁（不超过10个字）
2. 优先级根据讨论的热度和重要性判断
3. 总结要精炼，每句话不超过30字
4. record_intervals 应该尽量覆盖所有聊天记录（从1到{len(chat_records)}）
5. 区间可以有少量重叠，但要尽量减少
6. 请确保返回的是有效的JSON格式
7. 如果上传的文件没有时间，就默认为0000年00月00日"""

        try:
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的聊天记录分析师，擅长识别和总结讨论话题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            # 解析API响应
            content = response.choices[0].message.content
            topics_data = self._parse_api_response(content)

            # 验证区间覆盖并转换为记录内容（允许重叠）
            self._validate_and_convert_intervals(topics_data, chat_records, allow_overlap=True)

            # 更新聊天结构
            updated_structure = self._update_chat_structure(
                group_name, topics_data["topics"],
                existing_structure, description
            )

            return updated_structure
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")

    def _format_chat_records_with_indices(self, records: List[str]) -> str:
        """格式化聊天记录并添加索引"""
        formatted = []
        for i, record in enumerate(records, 1):
            formatted.append(f"[{i}] {record}")
        return "\n".join(formatted)

    def _format_existing_topics(self, topics: List[Dict]) -> str:
        """格式化已有话题信息"""
        if not topics:
            return "暂无已有话题"

        formatted = []
        for i, topic in enumerate(topics, 1):
            formatted.append(f"{i}. {topic['topic_name']} (优先级: {topic['priority']})")
            if topic['summaries']:
                formatted.append(f"   总结: {topic['summaries'][0]}")
            if topic['related_topics']:
                formatted.append(f"   相关话题: {', '.join(topic['related_topics'][:3])}")

        return "\n".join(formatted)

    def _parse_api_response(self, response_content: str) -> Dict:
        """解析API响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("API响应中没有找到有效的JSON格式")
        except json.JSONDecodeError as e:
            # 尝试修复常见的JSON格式问题
            try:
                # 移除可能的多余字符
                cleaned = response_content.strip()
                if cleaned.startswith('```json'):
                    cleaned = cleaned[7:]
                if cleaned.startswith('```'):
                    cleaned = cleaned[3:]
                if cleaned.endswith('```'):
                    cleaned = cleaned[:-3]

                return json.loads(cleaned)
            except:
                raise ValueError(f"无法解析API响应为JSON: {str(e)}\n响应内容: {response_content[:200]}...")

    def _validate_and_convert_intervals(self, topics_data: Dict, chat_records: List[str],
                                        allow_overlap: bool = True) -> None:
        """验证区间覆盖并转换为记录内容"""
        total_records = len(chat_records)

        # 收集所有区间
        all_intervals = []
        for topic in topics_data["topics"]:
            intervals = topic.get("record_intervals", [])
            if not intervals:
                raise ValueError(f"话题 '{topic.get('topic_name', '未知')}' 缺少 record_intervals 字段")

            # 验证区间格式
            valid_intervals = []
            for interval in intervals:
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(f"区间格式错误: {interval}，应为 [开始索引, 结束索引]")

                start, end = interval
                # 尝试转换为整数
                try:
                    start = int(start)
                    end = int(end)
                except (ValueError, TypeError):
                    raise ValueError(f"区间索引必须为整数: {interval}")

                if start < 1 or end > total_records:
                    raise ValueError(f"区间 {interval} 超出记录范围 (1-{total_records})")

                if start > end:
                    raise ValueError(f"区间起始索引不能大于结束索引: {interval}")

                valid_intervals.append([start, end])

            topic["record_intervals"] = valid_intervals
            all_intervals.extend([(start, end, i) for i, (start, end) in enumerate(valid_intervals)])

        if not allow_overlap:
            # 检查区间覆盖（不允许重叠）
            covered = [False] * total_records

            for start, end, _ in all_intervals:
                for i in range(start - 1, end):
                    if covered[i]:
                        # 查找是哪个区间覆盖了这个记录
                        for other_start, other_end, _ in all_intervals:
                            if other_start <= i + 1 <= other_end and (other_start, other_end) != (start, end):
                                raise ValueError(
                                    f"记录 {i + 1} 被多个区间覆盖: [{start}, {end}] 和 [{other_start}, {other_end}]")
                    covered[i] = True
        else:
            # 允许重叠，只检查每个记录是否至少被一个区间覆盖
            covered = [False] * total_records
            for start, end, _ in all_intervals:
                for i in range(start - 1, end):
                    covered[i] = True

        # 检查是否有未覆盖的记录
        uncovered = [i + 1 for i, is_covered in enumerate(covered) if not is_covered]
        if uncovered:
            print(
                f"警告: 有 {len(uncovered)} 条记录未被任何话题覆盖: {uncovered[:10]}{'...' if len(uncovered) > 10 else ''}")

            # 尝试将未覆盖的记录分配到现有话题
            self._assign_uncovered_records(topics_data, uncovered, chat_records)

        # 将区间转换为记录内容
        for topic in topics_data["topics"]:
            intervals = topic["record_intervals"]
            related_records = []

            for start, end in intervals:
                # 注意：start和end是从1开始的索引，需要转换为0-based索引
                related_records.extend(chat_records[start - 1:end])

            # 添加转换后的记录内容
            topic["related_records"] = related_records

            # 可选：保留原始区间信息
            topic["original_intervals"] = intervals

    def _assign_uncovered_records(self, topics_data: Dict, uncovered: List[int], chat_records: List[str]) -> None:
        """将未覆盖的记录分配到最近的话题"""
        if not topics_data["topics"] or not uncovered:
            return

        # 按索引排序
        uncovered.sort()

        # 为每个未覆盖的记录找到最近的话题
        for record_idx in uncovered:
            # 找到距离最近的话题区间
            min_distance = float('inf')
            best_topic_idx = 0
            best_position = "before"  # "before" 或 "after"

            for topic_idx, topic in enumerate(topics_data["topics"]):
                intervals = topic["record_intervals"]
                for start, end in intervals:
                    # 计算距离
                    if record_idx < start:
                        distance = start - record_idx
                        if distance < min_distance:
                            min_distance = distance
                            best_topic_idx = topic_idx
                            best_position = "before"
                    elif record_idx > end:
                        distance = record_idx - end
                        if distance < min_distance:
                            min_distance = distance
                            best_topic_idx = topic_idx
                            best_position = "after"
                    # 如果记录在区间内，应该已经被覆盖，这里不会出现

            # 将记录添加到最近的话题
            target_topic = topics_data["topics"][best_topic_idx]

            if best_position == "before":
                # 在现有区间前添加新区间或扩展第一个区间
                intervals = target_topic["record_intervals"]
                if intervals and intervals[0][0] == record_idx + 1:
                    # 可以扩展第一个区间
                    intervals[0][0] = record_idx
                else:
                    # 添加新区间
                    intervals.insert(0, [record_idx, record_idx])
            else:  # after
                # 在现有区间后添加新区间或扩展最后一个区间
                intervals = target_topic["record_intervals"]
                if intervals and intervals[-1][-1] == record_idx - 1:
                    # 可以扩展最后一个区间
                    intervals[-1][-1] = record_idx
                else:
                    # 添加新区间
                    intervals.append([record_idx, record_idx])

            # 更新话题的区间
            target_topic["record_intervals"] = intervals

    def _update_chat_structure(self, group_name: str, new_topics: List[Dict],
                               existing_structure: Optional[Dict],
                               description: str) -> Dict:
        """更新聊天结构"""
        if existing_structure:
            structure = existing_structure
        else:
            structure = {"chat_groups": []}
        # 查找是否已有该群聊
        group_id = None
        for group in structure["chat_groups"]:
            if group["group_name"] == group_name:
                group_id = group["group_id"]
                break
        # 如果没有该群聊，创建新的
        if not group_id:
            group_id = f"group_{len(structure['chat_groups']) + 1:03d}"
            new_group = {
                "group_id": group_id,
                "group_name": group_name,
                "description": description,
                "topics": []
            }
            structure["chat_groups"].append(new_group)
        # 更新话题
        for group in structure["chat_groups"]:
            if group["group_id"] == group_id:
                # 为每个新话题添加ID
                for topic in new_topics:
                    topic_id = f"topic_{group_id.replace('group_', '')}_{len(group['topics']) + 1:02d}"
                    topic["topic_id"] = topic_id
                    # 移除临时字段
                    if "record_intervals" in topic:
                        del topic["record_intervals"]
                    if "original_intervals" in topic:
                        del topic["original_intervals"]
                    group["topics"].append(topic)
                break

        self.chat_structure = structure
        return structure

    def generate_report(self, topic_id: str, report_type: str = "detailed") -> str:
        """
        为指定话题生成报告
        """
        # 查找话题信息
        topic_info = None
        group_info = None

        for group in self.chat_structure["chat_groups"]:
            for topic in group["topics"]:
                if topic["topic_id"] == topic_id:
                    topic_info = topic
                    group_info = group
                    break
            if topic_info:
                break

        if not topic_info:
            raise ValueError(f"未找到话题ID: {topic_id}")

        # 准备报告生成提示
        report_prompts = {
            "summary": "生成一个简洁的话题总结报告，包括主要讨论点和结论。",
            "detailed": "生成详细的分析报告，包括背景、讨论内容、关键成员、情绪、关键观点、结论和建议。",
            "analysis": "进行深入分析，包括趋势分析、观点对比、潜在问题和发展建议。"
        }

        prompt = f"""请根据以下话题信息，生成一份{report_type}报告。

群聊信息：
- 群聊名称：{group_info['group_name']}
- 群聊描述：{group_info.get('description', '暂无描述')}

话题信息：
- 话题名称：{topic_info['topic_name']}
- 优先级：{topic_info['priority']}
- 话题总结：{'；'.join(topic_info['summaries'])}
- 相关记录：{chr(10).join(topic_info['related_records'][:5])}
- 相关话题：{', '.join(topic_info['related_topics'])}

请生成一份{report_prompts[report_type]}
报告要求：
1. 结构清晰，逻辑连贯
2. 使用专业但不晦涩的语言
3. 基于提供的信息进行分析和总结
4. 如果可能，提供有价值的见解和建议
5. 不要有无关的字符比如**和##影响报告阅读

报告格式：
请使用纯文本格式，不要使用Markdown。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的报告撰写专家，擅长根据话题信息生成结构清晰、内容详实的报告。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"报告生成失败: {str(e)}")

    def save_structure(self, file_path: str):
        """保存聊天结构到JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.chat_structure, f, ensure_ascii=False, indent=2)

    def load_structure(self, file_path: str):
        """从JSON文件加载聊天结构"""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.chat_structure = json.load(f)

    def export_report(self, topic_id: str, output_path: str, report_type: str = "detailed"):
        """导出报告到文件"""
        report_content = self.generate_report(topic_id, report_type)

        # 添加标题和时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        topic_name = ""

        for group in self.chat_structure["chat_groups"]:
            for topic in group["topics"]:
                if topic["topic_id"] == topic_id:
                    topic_name = topic["topic_name"]
                    break

        full_report = f"""话题分析报告
====================

报告生成时间：{timestamp}
话题ID：{topic_id}
话题名称：{topic_name}
报告类型：{report_type}

{report_content}

--- 报告结束 ---"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)

        return output_path

    def analyze_topics_by_intervals(self, group_name: str, chat_records: List[str],
                                    existing_structure: Optional[Dict] = None,
                                    description: str = "") -> Dict:
        """
        分析聊天记录并生成话题结构（使用区间方法）
        这是analyze_topics的别名，保持向后兼容
        """
        return self.analyze_topics(group_name, chat_records, existing_structure, description)


def main():
    """示例运行函数"""
    API_KEY = "sk-570a56309e00406c961afb132ddb9053"

    # 初始化分析器
    analyzer = ChatAnalyzer(api_key=API_KEY)

    # 示例聊天记录（如果没有文件，可以使用文本列表）
    sample_records = [
        "2023-10-01 10:00 张三: 大家觉得这个新项目怎么样？",
        "2023-10-01 10:05 李四: 我觉得很有前景，但需要更多细节",
        "2023-10-01 10:10 王五: 我们需要先做市场调研",
        "2023-10-01 10:15 张三: 同意，市场调研很重要",
        "2023-10-01 10:20 李四: 我们可以下周开会讨论分工",
        "2023-10-01 10:25 王五: 好的，我准备会议材料",
        "2023-10-01 10:30 张三: 另外，关于预算的问题",
        "2023-10-01 10:35 李四: 预算需要详细规划",
        "2023-10-01 10:40 王五: 我们可以参考去年的项目",
        "2023-10-01 10:45 张三: 好的，我先做个初步预算"
    ]

    try:
        # 分析话题
        print("开始分析聊天记录...")
        structure = analyzer.analyze_topics(
            group_name="项目讨论组",
            chat_records=sample_records,
            description="新项目初步讨论"
        )

        # 保存结构
        analyzer.save_structure("chat_structure.json")
        print("聊天结构已保存到 chat_structure.json")

        # 显示话题信息
        print("\n发现的话题:")
        for group in structure["chat_groups"]:
            print(f"群聊: {group['group_name']}")
            for topic in group["topics"]:
                print(f"  - {topic['topic_name']} (优先级: {topic['priority']})")
                print(f"    记录数量: {len(topic['related_records'])}条")
                print(f"    总结: {'; '.join(topic['summaries'])}")

        # 如果有话题，生成报告
        if structure["chat_groups"] and structure["chat_groups"][0]["topics"]:
            topic_id = structure["chat_groups"][0]["topics"][0]["topic_id"]

            # 生成报告
            print(f"\n为话题 {topic_id} 生成报告...")
            report = analyzer.generate_report(topic_id, "summary")
            print("报告内容预览:")
            print(report[:500] + "..." if len(report) > 500 else report)

            # 导出报告
            analyzer.export_report(topic_id, "report.txt", "summary")
            print("报告已导出到 report.txt")

        print("\n分析完成！")

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("请确保：")
        print("1. 已安装所有依赖: pip install python-docx pdfplumber openai")
        print("2. 已设置正确的 API 密钥")
        print("3. 如果使用文件解析，确保文件路径正确")


if __name__ == "__main__":
    main()