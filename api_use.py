import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

# 文件解析相关库
import pdfplumber
import docx
import docx2txt

# API调用
from openai import OpenAI


class ChatAnalyzer:
    def __init__(self, api_key: str, base_url: str = "https://api-inference.modelscope.cn/v1/"):
        """
        初始化聊天记录分析器
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.chat_structure = {
            "chat_groups": []
        }

    def parse_file(self, file_path: str)->List[str]:
        """
        解析上传的文件，支持PDF、DOC、DOCX格式
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext=='.pdf':
            return self._parse_pdf(file_path)
        elif ext=='.docx':
            return self._parse_docx(file_path)
        elif ext=='.doc':
            return self._parse_doc(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，请上传PDF、DOC或DOCX文件")

    def _parse_pdf(self,file_path: str) -> List[str]:
        """解析PDF文件"""
        records = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text=page.extract_text()
                    if text:
                        # 按行分割并清理空白行
                        lines=[line.strip() for line in text.split('\n') if line.strip()]
                        records.extend(lines)
        except Exception as e:
            raise Exception(f"PDF解析失败: {str(e)}")

        return self._clean_and_limit_records(records)

    def _parse_docx(self,file_path: str) -> List[str]:
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

    def _parse_doc(self,file_path: str) -> List[str]:
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
                result=subprocess.run(['antiword', file_path], capture_output=True, text=True)
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

    def analyze_topics(self,group_name: str,chat_records:List[str],
                       existing_structure: Optional[Dict] = None,
                       description: str = "") -> Dict:
        """
        分析聊天记录并生成话题结构
        """
        if not chat_records:
            raise ValueError("聊天记录不能为空")

        # 准备现有的群聊话题信息
        existing_topics_info = ""
        if existing_structure and "chat_groups" in existing_structure:
            for group in existing_structure["chat_groups"]:
                if group["group_name"] == group_name:
                    existing_topics_info = self._format_existing_topics(group["topics"])
                    break

        # 格式化聊天记录
        formatted_records = self._format_chat_records(chat_records)

        # 准备API调用
        prompt = f"""你是一个专业的聊天记录分析师。请分析以下聊天记录，并识别出不同的讨论话题。

已有的群聊话题：
{existing_topics_info if existing_topics_info else "暂无已有话题"}

需要分析的聊天记录：
{formatted_records}

请完成以下任务：
1. 识别聊天记录中的主要讨论话题
2. 为每个话题生成：
   话题名称（简洁明了）
   优先级（高/中/低）
   简要总结（3-5句话）
   相关的聊天记录（选择3-5条最具代表性的记录）
   相关话题（与其他话题的关联）

请以严格的JSON格式返回，格式如下：
{{
  "topics": [
    {{
      "topic_name": "话题名称",
      "priority": "高/中/低",
      "summaries": ["总结1", "总结2", "总结3"],
      "related_records": ["记录1", "记录2", "记录3"],
      "related_topics": ["相关话题1", "相关话题2"]
    }}
  ]
}}

注意：
1. 话题名称要简洁（不超过10个字）
2. 优先级根据讨论的热度和重要性判断
3. 总结要精炼，每句话不超过30字
4. 相关记录从提供的聊天记录中选择
5. 相关话题可以从已有话题中选择，也可以创建新的
6. 请确保返回的是有效的JSON格式"""

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

            # 更新聊天结构
            updated_structure = self._update_chat_structure(
                group_name, topics_data["topics"],
                existing_structure, description
            )

            return updated_structure
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")

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

    def _format_chat_records(self, records: List[str]) -> str:
        """格式化聊天记录"""
        formatted = []
        for i, record in enumerate(records[:50], 1):
            formatted.append(f"{i}. {record}")

        if len(records) > 50:
            formatted.append(f"... 等{len(records)}条记录")

        return "\n".join(formatted)

    def _parse_api_response(self, response_content: str) -> Dict:
        """解析API响应"""
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*}', response_content, re.DOTALL)
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
                    group["topics"].append(topic)
                break

        self.chat_structure = structure
        return structure

    def generate_report(self, topic_id: str, report_type: str = "detailed") -> str:
        """
        为指定话题生成报告
        """
        # 查找话题信息
        topic_info=None
        group_info=None

        for group in self.chat_structure["chat_groups"]:
            for topic in group["topics"]:
                if topic["topic_id"]==topic_id:
                    topic_info=topic
                    group_info=group
                    break
            if topic_info:
                break

        if not topic_info:
            raise ValueError(f"未找到话题ID: {topic_id}")

        # 准备报告生成提示
        report_prompts={
            "summary":"生成一个简洁的话题总结报告，包括主要讨论点和结论。",
            "detailed":"生成详细的分析报告，包括背景、讨论内容、关键成员、情绪、关键观点、结论和建议。",
            "analysis":"进行深入分析，包括趋势分析、观点对比、潜在问题和发展建议。"
        }

        prompt=f"""请根据以下话题信息，生成一份{report_type}报告。

群聊信息：
- 群聊名称：{group_info['group_name']}
- 群聊描述：{group_info.get('description', '暂无描述')}

话题信息：
- 话题名称：{topic_info['topic_name']}
- 优先级：{topic_info['priority']}
- 话题总结：{'；'.join(topic_info['summaries'])}
- 相关记录：{chr(10).join(topic_info['related_records'][:3])}
- 相关话题：{', '.join(topic_info['related_topics'])}

请生成一份{report_prompts[report_type]}
报告要求：
1. 结构清晰，逻辑连贯
2. 使用专业但不晦涩的语言
3. 基于提供的信息进行分析和总结
4. 如果可能，提供有价值的见解和建议

报告格式：
请使用纯文本格式，不要使用Markdown。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content":"你是一个专业的报告撰写专家，擅长根据话题信息生成结构清晰、内容详实的报告。"},
                    {"role":"user","content": prompt}
                ],
                temperature=0.5,
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"报告生成失败: {str(e)}")

    def save_structure(self,file_path: str):
        """保存聊天结构到JSON文件"""
        with open(file_path,'w',encoding='utf-8') as f:
            json.dump(self.chat_structure,f,ensure_ascii=False, indent=2)

    def load_structure(self,file_path:str):
        """从JSON文件加载聊天结构"""
        with open(file_path,'r',encoding='utf-8') as f:
            self.chat_structure = json.load(f)

    def export_report(self,topic_id: str,output_path: str,report_type:str = "detailed"):
        """导出报告到文件"""
        report_content=self.generate_report(topic_id,report_type)

        # 添加标题和时间戳
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        topic_name=""

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