import time
import json
import os
from collections import defaultdict
from openai import OpenAI
from config import API_KEY
import pywechat
from queue import Queue
import threading


class WeChatGroupMonitor:
    def __init__(self, threshold=15):
        """
        初始化微信群聊监控器

        Args:
            threshold: 触发总结的消息数量阈值
        """
        self.threshold = threshold

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=API_KEY,
            base_url="https://api-inference.modelscope.cn/v1/"
        )

        # 存储群聊信息
        self.group_info = {}
        # 存储群聊消息 {群聊ID: [消息列表]}
        self.group_messages = defaultdict(list)
        # 允许监听的群聊列表
        self.allowed_groups = set()

        # 消息队列
        self.msg_queue = Queue()

        # 微信自动化实例
        self.w = None

        # 运行状态
        self.is_running = True

    def initialize_wechat(self):
        """初始化微信客户端"""
        print("正在初始化微信客户端...")
        try:
            self.w = pywechat.WeChat()
            print("微信客户端初始化成功！")
            return True
        except Exception as e:
            print(f"微信客户端初始化失败: {e}")
            return False

    def get_group_list(self):
        """获取并选择要监听的群聊"""
        print("\n请手动操作微信，将需要监听的群聊置顶")
        print("程序将自动检测置顶的群聊进行监听")
        input("按回车键继续...")

        # 在实际使用中，这里可能需要通过其他方式获取群聊列表
        # 由于pywechat的限制，我们让用户手动选择群聊

        print("\n请在接下来的步骤中选择要监听的群聊:")
        print("1. 程序将监听所有置顶的群聊")
        print("2. 或者您可以在代码中手动指定群聊ID")

        # 这里可以手动添加已知的群聊ID
        manual_groups = input("\n请输入要监听的群聊ID（多个用逗号分隔，直接回车跳过）: ").strip()

        if manual_groups:
            groups = [g.strip() for g in manual_groups.split(',')]
            self.allowed_groups.update(groups)
            print(f"已设置监听群聊: {', '.join(groups)}")
        else:
            print("将监听所有检测到的群聊消息")

    def message_handler(self, msg):
        """处理接收到的消息"""
        try:
            msg_type = msg.get("msg_type")
            wx_id = msg.get("wx_id")
            content = msg.get("content", "")

            # 只处理文本消息
            if msg_type == 1:  # 文本消息
                # 判断是否为群聊消息（群聊ID通常包含@chatroom）
                if "@chatroom" in wx_id:
                    self.process_group_message(msg)

            # 处理其他类型的消息（可选）
            elif msg_type == 3:  # 图片消息
                print(f"收到图片消息，来自: {wx_id}")
            elif msg_type == 34:  # 语音消息
                print(f"收到语音消息，来自: {wx_id}")
            elif msg_type == 37:  # 好友申请
                print("收到好友申请")
                # 可以自动同意好友申请
                # self.w.agree_friend(msg_data=msg)

        except Exception as e:
            print(f"处理消息时出错: {e}")

    def process_group_message(self, msg):
        """处理群聊消息"""
        group_id = msg.get("wx_id")
        sender = msg.get("sender", "未知用户")
        content = msg.get("content", "")
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # 如果指定了监听群聊列表，且当前群聊不在列表中，则忽略
        if self.allowed_groups and group_id not in self.allowed_groups:
            return

        # 初始化群聊信息
        if group_id not in self.group_info:
            self.group_info[group_id] = {
                'name': f"群聊_{group_id}",
                'last_active': timestamp
            }

        # 存储消息
        message_record = {
            'time': timestamp,
            'sender': sender,
            'content': content,
            'raw_msg': msg
        }

        self.group_messages[group_id].append(message_record)

        # 更新群聊最后活动时间
        self.group_info[group_id]['last_active'] = timestamp

        print(f"[{timestamp}] {self.group_info[group_id]['name']} - {sender}: {content}")

        # 检查是否达到阈值
        if len(self.group_messages[group_id]) >= self.threshold:
            print(f"\n=== 群聊 '{self.group_info[group_id]['name']}' 消息达到阈值({self.threshold}条)，开始总结 ===")

            # 在新线程中处理总结，避免阻塞消息接收
            summary_thread = threading.Thread(
                target=self.process_summary,
                args=(group_id,)
            )
            summary_thread.daemon = True
            summary_thread.start()

    def process_summary(self, group_id):
        """处理话题总结"""
        try:
            # 调用大模型总结
            summary = self.call_llm_api(group_id, self.group_messages[group_id])

            if summary:
                # 保存总结和原始消息
                filename = self.save_summary(group_id, summary, self.group_messages[group_id])

                # 可选：将总结发送到文件传输助手
                try:
                    preview = f"【{self.group_info[group_id]['name']}话题总结】\n{summary[: