from openai import OpenAI
##要安装库
client = OpenAI(
    api_key="ms-f940f240-c625-4f7a-bbfb-5388db925ec7", # 请替换成您的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn/v1/"
)


response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-32B-Instruct", # ModelScope Model-Id
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '用python写一下快排'
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)