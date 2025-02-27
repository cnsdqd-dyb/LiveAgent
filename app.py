import streamlit as st
import json
import time
from PIL import Image
import os
import numpy as np
import cv2
from multiprocessing import shared_memory
from streamlit_autorefresh import st_autorefresh
import re

# 清理共享内存的函数
def cleanup_shared_memory():
    try:
        shared_memory.SharedMemory(name='frame_share').unlink()
        shared_memory.SharedMemory(name='frame_shape').unlink()
    except:
        pass

# 在应用启动时清理共享内存
cleanup_shared_memory()

st.set_page_config(
    page_title="A.L.I.C.E", # A.L.I.C.E.
    page_icon="🪄",
    layout="wide"
)

# 设置页面样式
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        border-radius: 30px;
        background: linear-gradient(to right, #0066ff, #00ccff);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 204, 255, 0.5);
    }
    .sidebar {
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.7);
    }
    .main {
        padding: 2rem;
    }
    .status-box {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 204, 255, 0.2);
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #00ccff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    body {
        background: url("https://i.imgur.com/your-tech-background.gif");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }
    .tech-container {
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(0, 204, 255, 0.3);
    }
    .neon-border {
        box-shadow: 0 0 10px #00ccff, 0 0 20px #00ccff, 0 0 30px #00ccff;
    }
    .chat-container {
        height: 600px;
        overflow-y: auto;
        padding: 10px;
    }
    .image-container {
        margin-top: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'show_full_history' not in st.session_state:
    st.session_state.show_full_history = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0

# 侧边栏信息显示
with st.sidebar:
    st.markdown("### 🪄 A.L.I.C.E 系统状态")
    
    with st.expander("📊 系统配置", expanded=True):
        st.markdown("""
        - **核心模型**: `phi4:latest`
        - **神经网络容量**: 4096 tokens
        - **系统状态**: ✅ 在线运行中
        """)
    
    with st.expander("🔄 实时监测", expanded=True):
        try:
            with open("image_description.json", "r", encoding='utf-8') as f:
                vision_info = json.load(f)
                st.markdown("##### 👁️ 视觉识别")
                for face in vision_info.get("faces", []):
                    st.markdown(f"- _{face}_")
                st.markdown("##### 🎯 场景分析")
                st.info(vision_info.get("caption", "环境扫描中..."))
        except:
            st.warning("📸 视觉系统待机中")

        try:
            with open("detail.json", "r", encoding='utf-8') as f:
                detail = json.load(f)
                st.markdown("##### 💡 神经网络记忆")
                st.info(detail.get("doc_content", "记忆检索中..."))
                st.markdown("##### ⚡ 最新指令")
                st.info(detail.get("last_input", "等待指令..."))
        except:
            st.warning("💫 系统初始化中...")

# 主界面
st.markdown("<h1 style='text-align: center; color: #00ccff; margin-top: -2rem;'>🪄 A.L.I.C.E 🪄</h1>", unsafe_allow_html=True)

# 创建两列布局
left_column, right_column = st.columns([1.2, 0.8])
try:
    with open("detail.json", "r", encoding='utf-8') as f:
        detail = json.load(f)
        loading = detail.get("loading", False)
except:
    loading = False

# 控制按钮
if loading:
    st.info("🔄 正在加载中，请稍等。。。")
else:
    with left_column:
        # 对话区域
        st.markdown("### 对话记录")
        with st.container():
            # 历史对话控制按钮
            if st.button("📜 " + ("隐藏历史记录" if st.session_state.show_full_history else "显示完整历史")):
                st.session_state.show_full_history = not st.session_state.show_full_history
                st.rerun()
            # 显示对话历史
            try:
                with open("detail.json", "r", encoding='utf-8') as f:
                    detail = json.load(f)
                    messages = detail.get("conversation_history", [])
                    if st.session_state.show_full_history:
                        display_messages = messages
                    else:
                        display_messages = messages[-3:]
                    
                    for msg in display_messages:
                        if msg["role"] == "user":
                            with st.chat_message("user", avatar="🧑🏻‍💼"):
                                content = msg["content"].split("语音识别检测到: ")[1][:-1]
                                st.markdown(f"**{content}**")
                        else:
                            with st.chat_message("assistant", avatar="🧙🏻"):
                                content = msg["content"]
                                # 去掉 <> 标签
                                content = re.sub(r'<[^>]+>', '', content)
                                # 去掉 [] 标签
                                content = re.sub(r'\[[^]]+\]', '', content)
                                st.markdown(content)
                    
                    loading = detail.get("loading", False)
            except:
                st.info("系统已就绪，Sir。")

            if json.load(open("interrupt.json", "r"))["interrupt"] == False:
                if st.button("🛑 中断并继续输入指令，Sir", use_container_width=True):
                    with open("interrupt.json", "w") as f:
                        f.write('{"interrupt": true}')
                    st.success("🛑 中断成功")
            else:
                st.button("🎙️等待您的回应，Sir", use_container_width=True)

    with right_column:
        st.markdown("### 实时渲染")
        
        # 创建两列：一列显示视频，一列显示音量条
        video_col, volume_col = st.columns([0.8, 0.2])
        
        with video_col:
            # 视频显示相关的代码保持不变
            if 'image_placeholder' not in st.session_state:
                st.session_state.image_placeholder = st.empty()
            
            if 'last_update' not in st.session_state:
                st.session_state.last_update = time.time()

            try:
                shape_shm = shared_memory.SharedMemory(name='frame_shape')
                shape_str = bytes(shape_shm.buf).decode().strip('\x00')
                height, width, channels = map(int, shape_str.split(','))
                frame_shape = (height, width, channels)

                shm = shared_memory.SharedMemory(name='frame_share')
                shared_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
                
                frame = shared_frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                st.session_state.image_placeholder.image(
                    image, 
                    caption="实时监控", 
                    use_container_width=True
                )
                    
            except FileNotFoundError:
                st.session_state.image_placeholder.warning("等待视频流连接...")
            except Exception as e:
                st.session_state.image_placeholder.error(f"视频流错误: {str(e)}")
                print(e)

        with volume_col:
            # 创建音量条的占位符
            if 'volume_placeholder' not in st.session_state:
                st.session_state.volume_placeholder = st.empty()
            
            try:
                # 读取 mouth_rate 值
                with open('live2d_mouth.json', 'r') as f:
                    mouth_data = json.load(f)
                mouth_rate = mouth_data.get('mouth_rate', 0)
                
                # 将 mouth_rate 映射到 0-100 的范围
                volume_height = int(mouth_rate * 100)
                
                # 根据音量大小确定颜色
                if volume_height < 30:
                    color = "green"
                elif volume_height < 70:
                    color = "orange"
                else:
                    color = "red"
                
                # 创建音量条的HTML
                volume_html = f"""
                <div style="
                    height: 500px;
                    width: 30px;
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    position: relative;
                    margin: auto;
                ">
                    <div style="
                        position: absolute;
                        bottom: 0;
                        width: 100%;
                        height: {volume_height}%;
                        background-color: {color};
                        border-radius: 3px;
                        transition: all 0.2s ease;
                    "></div>
                </div>
                <div style="
                    text-align: center;
                    margin-top: 10px;
                    font-size: 12px;
                ">
                    音量: {volume_height}%
                </div>
                """
                
                # 更新音量条显示
                st.session_state.volume_placeholder.markdown(volume_html, unsafe_allow_html=True)
                
            except FileNotFoundError:
                st.session_state.volume_placeholder.warning("未找到音量数据")
            except Exception as e:
                st.session_state.volume_placeholder.error(f"音量显示错误: {str(e)}")


count = st_autorefresh(interval=300, limit=None, debounce=False, key="fizzbuzzcounter")
