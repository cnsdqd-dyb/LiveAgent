from asr import VoiceListener
from vision_loop import VideoAnalyzer
from tts import AudioTTS
from llm import OllamaChat
from live_model import Live2DController
from rag import DocumentIndexer
import time
import json
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
# 初始化
with open("audio.json", "w") as f:
    f.write('{"listening": false}')

# 初始化 # streamlit 中显示的 detail.json
with open("detail.json", "w") as f:
    f.write('{"doc_content": "", "last_input": "", "conversation_history": [], "loading": true}')

analyzer = VideoAnalyzer()
live_control = Live2DController()
# 创建文档索引器
indexer = DocumentIndexer(
    index_path="memory", 
    embedding_model="BAAI/bge-small-zh-v1.5"
)
chat = OllamaChat(
    model_name="phi4:latest", # 需要显示在 streamlit 上的模型名称
    embed_model="mxbai-embed-large",
    max_context_length=4096, # 需要显示在 streamlit 上的模型最大上下文长度
    tts=AudioTTS("40293f990bec4378957a155b477c59b1", live_control=live_control),
    live_control=live_control,
    online=True, # 需要显示在 streamlit 上的是否启用在线模型
)

voice_listener = VoiceListener(model_dir="asr_model/SenseVoiceSmall", device="cuda")
caption = ""
faces = []
centers = []

# 姓名map 需要显示在 streamlit 上的姓名映射
name_map = {
    "dongyubo": "董玉博",
}
with open("interrupt.json", "w") as f:
    f.write('{"interrupt": true}')
# update loading status
with open("detail.json", "w") as f:
    f.write('{"doc_content": "", "last_input": "", "conversation_history": [], "loading": false}')
# 需要显示在 streamlit 上实现的循环体
while True:
    time.sleep(0.1)
    if voice_listener.text_buffer:
        user_input = voice_listener.text_buffer.pop(0)
        with open("detail.json", "w", encoding='utf-8') as f:
            details = {
                "doc_content": "...",
                "last_input": user_input,
                "conversation_history": chat.conversation_history,
                "loading": False
            }
            f.write(json.dumps(details, ensure_ascii=False, indent=2))
        with open("interrupt.json", "w") as f:
            f.write('{"interrupt": false}')
        # 处理输入
        try:
            # 加载 image_description.json
            with open("image_description.json", "r", encoding='utf-8') as f:
                results = json.load(f)
                caption = results.get("caption", "")
                faces = results.get("faces", [])
        except Exception as e:
            faces = []
            caption = ""
            pass
        names = "".join([name_map.get(face, "") for face in faces])
        # 添加到索引
        doc_result = indexer.search(user_input, k=1)[0]
        doc_content = doc_result.page_content
        # update detail.json
        with open("detail.json", "w", encoding='utf-8') as f:
            details = {
                "doc_content": doc_content,
                "last_input": user_input,
                "conversation_history": chat.conversation_history,
                "loading": False
            }
            f.write(json.dumps(details, ensure_ascii=False, indent=2))
        # 需要显示在 streamlit 上分别显示 caption, doc_content, names, user_input
        input = f"目前，视觉信息看到{caption}，查询系统反馈{doc_content}，人脸识别检测出是{names}\n 语音识别检测到: {user_input}."
        content = chat.chat(input, stream=True)

        # update detail.json
        with open("detail.json", "w", encoding='utf-8') as f:
            details = {
                "doc_content": doc_content,
                "last_input": user_input,
                "conversation_history": chat.conversation_history,
                "loading": False
            }
            f.write(json.dumps(details, ensure_ascii=False, indent=2))

        # 添加到索引
        if content:
            indexer.add_text(user_input)
            indexer.add_text(content)
        # 计算content长度来决定等待时间
        if content:
            max_wait_time = 1.5 * len(content)
        else:
            max_wait_time = 0.1
        while max_wait_time > 0:
            time.sleep(0.1)
            max_wait_time -= 0.1
            if chat.tts.audio_queue.empty():
                break
        chat.tts.interrupt = True

        # 播放完成
        voice_listener.reset()
        
    else:
        voice_listener.off = False
        voice_listener
