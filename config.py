API_KEY = "ms-329e7a2e-5d6a-4058-844f-5550e79e93f1"
# config.py
# ModelScope API配置

ONEBOT_WS_URL = "ws://localhost:6700"  # OneBot WebSocket地址
ONEBOT_HTTP_URL = "http://localhost:5700"  # OneBot HTTP API地址

# 监控配置
MESSAGE_THRESHOLD = 15  # 触发总结的消息阈值
SAVE_INTERVAL = 60  # 数据保存间隔（秒）

# 群聊简介模板（可根据实际需求修改）
GROUP_DESCRIPTION_TEMPLATE = {
    "default": "这是一个普通的群聊讨论组",
    "tech": "技术交流群，主要讨论编程、开发和新技术",
    "game": "游戏交流群，分享游戏心得和组队",
    "study": "学习群，交流学习经验和资料分享"
}