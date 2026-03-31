import asyncio
import base64
import logging
import os
import re
import subprocess
import sys
import uuid
import tempfile
import gradio as gr
from pathlib import Path
from services.asr_service import get_asr_service
from services.llm_service import get_llm_service
from services.tts_service import get_tts_service
from services.crm_service import get_crm_service
from typing import Callable, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

asr_service = get_asr_service()
llm_service  = get_llm_service()
tts_service  = get_tts_service()
crm_service  = get_crm_service()
logger.info("✅ 所有后端服务初始化成功")

BASE_DIR = Path(__file__).parent

def load_b64(filename: str, mime: str) -> str:
    p = BASE_DIR / filename
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode()
        logger.info(f"✅ 图片加载: {filename} ({len(data)//1024} KB)")
        return f"data:{mime};base64,{data}"
    logger.warning(f"⚠️  未找到: {filename}")
    return ""

CHAR_SRC = load_b64("1.png",       "image/webp")
BG_SRC   = load_b64("beijing.png", "image/jpeg")

# ===================== 核心逻辑 =====================
async def handle_text_chat(user_msg: str, history: list, session_id: str):
    """
    处理文本聊天（增强版）
    
    【改进流程】
    原版：用户消息 → LLM（无上下文）→ 回复 → 提取用户信息
    改进：用户消息 → 提取关键词 → 搜索CRM → 构建上下文 → LLM（有知识库+CRM数据）→ 回复 → 提取用户信息
    """
    if not user_msg.strip():
        return history, None, ""
    
    # 1. 保存用户消息到 CRM
    crm_service.save_message(session_id, "user", user_msg)
    
    # 2. 构建对话历史
    llm_history = []
    for u, b in history:
        if u: llm_history.append({"role": "user",      "content": u})
        if b: llm_history.append({"role": "assistant", "content": b})
    
    # 3. 【新增】构建 CRM 上下文
    crm_context = {}
    
    # 3a. 获取当前用户的 CRM 信息（让 AI 知道在和谁对话）
    user_info_text = crm_service.get_user_text(session_id)
    if user_info_text:
        crm_context["user_info_text"] = user_info_text
        logger.info(f"👤 已加载当前用户信息: {user_info_text[:50]}...")
    
    # 3b. 从用户消息中提取关键词，搜索 CRM 中的客户
    keywords = llm_service._extract_search_keywords(user_msg)
    crm_search_results = []
    for kw in keywords:
        results = crm_service.search_users(kw)
        crm_search_results.extend(results)
    
    if crm_search_results:
        # 去重
        seen = set()
        unique_results = []
        for r in crm_search_results:
            key = r.get("session_id", "")
            if key and key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        # 格式化搜索结果
        search_lines = []
        for r in unique_results:
            parts = []
            if r.get("name"): parts.append(f"姓名：{r['name']}")
            if r.get("phone"): parts.append(f"电话：{r['phone']}")
            if r.get("email"): parts.append(f"邮箱：{r['email']}")
            if r.get("company"): parts.append(f"公司：{r['company']}")
            if r.get("needs"): parts.append(f"需求：{r['needs']}")
            if r.get("other"): parts.append(f"其他：{r['other']}")
            search_lines.append("  - " + "，".join(parts))
        
        crm_context["crm_search_result"] = "找到以下客户信息：\n" + "\n".join(search_lines)
        logger.info(f"🔍 CRM 搜索到 {len(unique_results)} 条客户信息")
    
    # 3c. 如果用户问的是"有哪些客户"、"所有客户"等，加载全部客户列表
    all_customer_keywords = ["所有客户", "有哪些客户", "客户列表", "全部客户", "查看客户", "客户信息", "有哪些用户"]
    if any(kw in user_msg for kw in all_customer_keywords):
        all_users = crm_service.get_all_users_text()
        crm_context["all_users_text"] = all_users
        logger.info("📋 已加载全部客户列表")
    
    # 4. 调用 LLM（传入 CRM 上下文 + 知识库）
    ai_reply = await llm_service.chat(user_msg, llm_history, crm_context=crm_context)
    
    # 4b. 清洗 Markdown 格式（Gradio 不渲染 Markdown，** 和 ## 会原样显示）
    ai_reply = re.sub(r'\*\*(.+?)\*\*', r'\1', ai_reply)       # **加粗** → 加粗
    ai_reply = re.sub(r'(?m)^#{1,6}\s+', '', ai_reply)          # ## 标题 → 去掉
    #ai_reply = re.sub(r'(?m)^[-*]\s+', '', ai_reply)             # - 列表 → 去掉符号
    #ai_reply = re.sub(r'`{1,3}[^`]*`{1,3}', '', ai_reply)       # `代码` → 去掉
    ai_reply = re.sub(r'\n{3,}', '\n\n', ai_reply)               # 多个空行 → 最多两个
    # 5. 保存 AI 回复到 CRM
    crm_service.save_message(session_id, "assistant", ai_reply)
    
    # 6. 从对话中提取用户信息并更新 CRM
    try:
        conv = crm_service.get_session_text(session_id)
        info = await llm_service.extract_user_info(conv)
        if info:
            crm_service.update_user_info(session_id, info)
            logger.info(f"📝 用户信息更新: {info}")
    except Exception as e:
        logger.warning(f"⚠️ 提取失败: {e}")
    
    # 7. 语音播报
    tts_text = re.sub(r"\([^)]*\)", " ", ai_reply)
    tts_text = re.sub(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]", " ", tts_text)
    tts_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", tts_text)
    tts_text = re.sub(r"\s+", " ", tts_text).strip()
    # 截断保护：超过 250字只取前 250字，语音合成太长会卡很久
    if len(tts_text) > 250:
        tts_text = tts_text[:250].rsplit(" ", 1)[0]  # 在最后一个空格处截断，避免断词
        logger.info(f"✂️ TTS 文本过长，已截断至 {len(tts_text)} 字")
    if not tts_text:
        tts_text = ai_reply
    tts_audio = await tts_service.synthesize(tts_text)
    audio_path = None
    if tts_audio:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(tts_audio)
            audio_path = f.name
    
    return history + [(user_msg, ai_reply)], audio_path, ""

async def handle_voice_input(audio_path: str, history: list, session_id: str):
    """接收音频文件路径，进行 ASR 识别后转文字聊天"""
    if not audio_path:
        return history, None, tip("⚠️ 未检测到录音")
    
    try:
        audio_bytes = open(audio_path, "rb").read()
        
        # 根据文件后缀判断格式，传给正确的转换方法
        if audio_path.endswith(".mp4"):
            user_text = await asr_service.transcribe_from_browser(audio_bytes, mime_type="mp4")
        elif audio_path.endswith(".webm"):
            user_text = await asr_service.transcribe_from_browser(audio_bytes, mime_type="webm")
        else:
            user_text = await asr_service.transcribe_from_browser(audio_bytes, mime_type="webm")
        
        if not user_text:
            return history, None, tip("⚠️ 未识别到语音，请重试")
        
        h, ap, _ = await handle_text_chat(user_text, history, session_id)
        return h, ap, tip("✅ 语音识别完成")
    except Exception as e:
        logger.error(f"❌ 语音失败: {e}")
        return history, None, tip(f"❌ 出错: {str(e)[:60]}")

# ── 新增：接收 base64 音频数据，保存临时文件后调用 handle_voice_input ──
def receive_voice_b64(b64_audio: str, history: list, session_id: str):
    """
    由前端 JS 通过隐藏 Textbox 传入 base64 编码的 webm 音频，
    解码后写入临时文件，再走 ASR 流程。
    """
    if not b64_audio or not b64_audio.strip():
        return history, build_chat_html(history), None, tip("⚠️ 未收到音频数据")
    try:
        audio_bytes = base64.b64decode(b64_audio)
        suffix = ".webm" if b64_audio[:20].find("webm") != -1 else ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        logger.info(f"🎙️ 收到录音: {len(audio_bytes)//1024} KB → {tmp_path}")
        nh, ap, s = asyncio.run(handle_voice_input(tmp_path, history, session_id))
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return nh, build_chat_html(nh), ap, s   # 不输出 voice_b64_input，由 JS 端清空
    except Exception as e:
        logger.error(f"❌ 录音处理失败: {e}")
        return history, build_chat_html(history), None, tip(f"❌ 出错: {str(e)[:60]}")

def wrap_async(func: Callable) -> Callable:
    def sync(*args: Any, **kwargs: Any):
        return asyncio.run(func(*args, **kwargs))
    return sync

def tip(text: str) -> str:
    return f'<p class="s-tip">{text}</p>'

def build_chat_html(history: list) -> str:
    if not history:
        return '<p class="empty-chat" style="margin-top: 20px;">✨ 开始对话吧～</p>'
    
    html = ""
    for u, b in history:
        if u:
            s = str(u).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html += f'<div class="row-u"><div class="bbl user-bbl">{s}</div></div>'
        if b:
            s = str(b).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            html += f'<div class="row-b"><div class="bbl bot-bbl">{s}</div></div>'
    
    html += """
    <script>
        var x = document.getElementById("cb");
        if(x) {
            x.scrollTo({ top: x.scrollHeight, behavior: 'smooth' });
        }
    </script>
    """
    return html

# ===================== CSS =====================
CSS = """
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
body, html { height:100vh; overflow:hidden; }

.gradio-container {
    max-width:100% !important; height:100vh !important;
    padding:0 !important; margin:0 !important;
    background:transparent !important; overflow:hidden !important;
}
footer, .built-with { display:none !important; }

/* ── 顶层行 ── */
.app-row {
    display:flex !important; flex-wrap:nowrap !important;
    width:100vw !important; height:100vh !important;
    position:relative; overflow:hidden; gap:0 !important;
    padding:0 12px !important;
    justify-content:center !important;
}

/* ── 全屏背景 ── */
.app-bg {
    position:fixed !important; inset:0;
    background-size:cover !important; background-position:center !important;
    z-index:0; filter:brightness(0.82); pointer-events:none;
}

/* ── 左侧角色区 ── */
.char-col {
    flex:0 0 50% !important; max-width:50% !important;
    position:relative; z-index:1;
    display:flex !important; flex-direction:column !important;
    align-items:center !important; justify-content:center !important;
    padding:0 !important; gap:0 !important;
    background:transparent !important; border:none !important; box-shadow:none !important;
    overflow:hidden;
}

.char-img {
    display:block;
    height:80vh; max-height:80vh; width:auto;
    object-fit:contain;
    margin-bottom:22px;
    animation:float 4s ease-in-out infinite;
    filter:drop-shadow(0 8px 28px rgba(120,60,255,0.55));
    position:relative; z-index:1;
}
@keyframes float {
    0%,100%{ transform:translateY(0); }
    50%    { transform:translateY(-12px); }
}

/* 状态标签 */
.status-badge {
    position:absolute; top:18px; left:20px; z-index:20;
    background:rgba(0,0,0,0.62); backdrop-filter:blur(10px);
    border:1px solid rgba(168,85,247,0.45); border-radius:20px;
    padding:5px 13px; color:#d8b4fe; font-size:12px;
    display:flex; align-items:center; gap:6px; white-space:nowrap;
}
.dot-live {
    width:7px; height:7px; border-radius:50%; background:#4ade80; flex-shrink:0;
    animation:blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.7)} }

/* ── 底部控制栏：图像正下方，向左偏移 20px ── */
.ctrl-bar {
    position:relative !important; z-index:20;
    display:flex !important; flex-direction:row !important;
    gap:14px !important; justify-content:flex-start !important;
    padding:12px 16px 13px !important;
    margin:-10px 0 24px -270px !important;
    background:linear-gradient(135deg, rgba(26,26,38,0.80), rgba(34,28,48,0.72)) !important;
    border:1px solid rgba(255,255,255,0.14) !important;
    border-radius:26px !important;
    box-shadow:0 10px 30px rgba(0,0,0,0.22) !important;
    backdrop-filter:blur(10px);
    width:auto !important;
}

/* 强制圆形按钮 */
.ctrl-bar button, .ctrl-bar .gr-button {
    width:66px !important; height:66px !important;
    min-width:66px !important; max-width:66px !important;
    min-height:66px !important; max-height:66px !important;
    border-radius:50% !important;
    padding:0 0 10px 0 !important; font-size:22px !important; line-height:1 !important;
    border:none !important; outline:none !important; cursor:pointer !important;
    display:inline-flex !important; align-items:center !important; justify-content:center !important;
    transition:transform 0.18s, box-shadow 0.18s !important; flex-shrink:0 !important;
    position:relative !important;
}
.ctrl-bar button:hover { transform:translateY(-1px) scale(1.05) !important; }
.ctrl-bar button::after {
    position:absolute;
    left:50%;
    bottom:6px;
    transform:translateX(-50%);
    font-size:10px;
    font-weight:700;
    color:#ffffff;
    opacity:0.96;
    white-space:nowrap;
    letter-spacing:0.6px;
}

#btn-s { background:linear-gradient(135deg,#4f46e5,#7c3aed) !important; box-shadow:0 6px 18px rgba(79,70,229,0.45) !important; color:#fff !important; }
#btn-e { background:linear-gradient(135deg,#ef4444,#f97316) !important; box-shadow:0 6px 18px rgba(239,68,68,0.44) !important; color:#fff !important; }
#btn-m { background:linear-gradient(135deg,#f59e0b,#fb7185) !important; box-shadow:0 6px 18px rgba(245,158,11,0.40) !important; color:#fff !important; }
#btn-r { background:linear-gradient(135deg,#a855f7,#7c3aed) !important; color:#fff !important; animation:prec 2s infinite !important; }
#btn-r.recording { background:#ef4444 !important; animation:none !important; box-shadow:0 4px 16px rgba(239,68,68,0.60) !important; }
#btn-s::after { content:"数据"; }
#btn-e::after { content:"摄像头"; }
#btn-m::after { content:"挂断"; }
#btn-r::after { content:"录音"; }
#btn-r[data-label]::after { content:attr(data-label); }
@keyframes prec {
    0%,100%{ box-shadow:0 4px 14px rgba(168,85,247,0.55); }
    50%    { box-shadow:0 4px 26px rgba(168,85,247,0.95), 0 0 0 9px rgba(168,85,247,0.13); }
}
/* ── 右侧聊天列整体左移 ── */
.chat-col {
        transform: translateX(-80px) !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
}
/* ── 右侧聊天区：28% 宽 ── */
/* 消息区 */
#cb {
    flex: 1 1 0 !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    padding: 20px 15px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    min-height: 0 !important;
}

/* 滚动条样式 */
#cb::-webkit-scrollbar { width: 8px !important; }
#cb::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.15) !important; }
#cb::-webkit-scrollbar-thumb {
    background: #a855f7 !important;
    border-radius: 4px !important;
    box-shadow: 0 0 6px rgba(168, 85, 247, 0.5) !important;
}
#cb::-webkit-scrollbar-thumb:hover { background: #c084fc !important; }

.empty-chat { 
    text-align: center;
    color: rgba(255,255,255,0.4); 
    font-size: 14px; 
    margin-top: 30px !important;
    font-style: italic;
}
.row-u { display:flex; justify-content:flex-end; margin-bottom:8px; }
.row-b { display:flex; justify-content:flex-start; margin-bottom:8px; }
.bbl   { max-width:84%; padding:9px 13px; font-size:16px; line-height:1.6; word-break:break-word; }
.user-bbl {
    background:linear-gradient(135deg,#c026d3,#7c3aed);
    color:#fff; border-radius:16px 16px 3px 16px;
    box-shadow:0 3px 10px rgba(124,58,237,0.38);
}
.bot-bbl {
    background:rgba(14,8,32,0.86);
    border:1px solid rgba(168,85,247,0.30);
    color:#e9d5ff; border-radius:16px 16px 16px 3px;
    box-shadow:0 3px 10px rgba(0,0,0,0.30); backdrop-filter:blur(6px);
}

/* 隐藏音频输出波形，只保留播放能力 */
.audio-zone {
    position:absolute !important;
    left:-9999px !important;
    top:auto !important;
    width:1px !important;
    height:1px !important;
    overflow:hidden !important;
    padding:0 !important;
    margin:0 !important;
    background:transparent !important;
    border:none !important;
    box-shadow:none !important;
}
.audio-zone .waveform-container,
.audio-zone canvas,
.audio-zone .waveform { display:none !important; }
.audio-zone audio { width:100%; height:28px; border-radius:14px; }

/* 状态提示 */
.s-tip {
    text-align:center; color:#000000; font-size:11px; padding:2px 10px; margin:0; flex-shrink:0;
    height:var(--chat-tip-h) !important;
    min-height:var(--chat-tip-h) !important;
    max-height:var(--chat-tip-h) !important;
    overflow:hidden !important;
    white-space:nowrap !important;
    text-overflow:ellipsis !important;
}

/* 隐藏 Gradio 内置加载 */
.gradio-container .time,
.gradio-container .loading-time,
.gradio-container .gradio-loading-time { display:none !important; }
.gradio-container [class*="status-tracker"],
.gradio-container [class*="progress-text"],
.gradio-container [class*="loading-text"],
.gradio-container [class*="eta"],
.gradio-container [class*="pending"] { display:none !important; }

/* 完全隐藏录音数据传输用的隐藏 Textbox */
.voice-hidden-zone {
    position:absolute !important;
    left:-9999px !important;
    top:auto !important;
    width:1px !important;
    height:1px !important;
    overflow:hidden !important;
    padding:0 !important;
    margin:0 !important;
    opacity:0 !important;
    pointer-events:none !important;
}

/* 输入区 */
.input-zone {
    flex-shrink: 0 !important;
    display:flex !important; flex-direction:row !important; gap:7px !important;
    padding:8px 10px 10px !important; flex-shrink:0 !important;
    border-top:none !important;
    align-items:center !important;
    margin-bottom: 40px !important;
    height:var(--chat-input-h) !important;
    min-height:var(--chat-input-h) !important;
    max-height:var(--chat-input-h) !important;
    position:relative !important;
    z-index:80 !important;
    backdrop-filter:none;
    background:transparent !important; border-left:none !important;
    border-right:none !important; border-bottom:none !important; box-shadow:none !important;
}
.input-zone textarea, .input-zone input[type="text"] {
    flex:1 !important;
    background:rgba(12,14,28,0.78) !important;
    border:1px solid rgba(255,255,255,0.22) !important;
    border-radius:16px !important; padding:10px 14px !important;
    color:#f0e8ff !important; font-size:13px !important;
    resize:none !important;
    height:44px !important;
    min-height:44px !important;
    max-height:44px !important;
    overflow-y:auto !important;
    line-height:1.25 !important;
    box-shadow:0 8px 22px rgba(0,0,0,0.22) !important;
}
.input-zone .gr-textbox,
.input-zone .gr-textbox > div,
.input-zone .gr-textbox .wrap,
.input-zone .gr-textbox label,
.input-zone .gr-textbox .wrap textarea {
    background:transparent !important;
    border-color:rgba(168,85,247,0.40) !important;
}
.input-zone .gr-textbox .wrap textarea {
    background:rgba(12,14,28,0.78) !important;
    color:#f0e8ff !important;
}
.input-zone textarea:disabled, .input-zone input[type="text"]:disabled {
    background:rgba(16,9,38,0.90) !important;
    color:#d8c8ff !important;
    -webkit-text-fill-color:#d8c8ff !important;
    opacity:0.72 !important;
}
.input-zone textarea:focus, .input-zone input:focus {
    border-color:#a855f7 !important;
    box-shadow:0 0 0 2px rgba(168,85,247,0.22) !important; outline:none !important;
}
#btn-send {
    width:40px !important; height:40px !important; min-width:40px !important; max-width:40px !important;
    border-radius:50% !important;
    background:linear-gradient(135deg,#c026d3,#7c3aed) !important;
    color:#fff !important; font-size:15px !important; padding:0 !important;
    border:none !important; cursor:pointer !important; flex-shrink:0 !important;
    box-shadow:0 3px 10px rgba(124,58,237,0.42) !important;
    display:inline-flex !important; align-items:center !important; justify-content:center !important;
    transition:transform 0.15s !important;
}
#btn-send:hover { transform:scale(1.10) !important; }

/* 全局隐藏 Gradio 任务遮罩 */
.gradio-container [class*="loading"],
.gradio-container [class*="spinner"],
.gradio-container [class*="overlay"],
.gradio-container [class*="skeleton"],
.gradio-container [class*="pending"],
.gradio-container [class*="progress"],
.gradio-container [data-testid*="status"],
.gradio-container [data-testid*="progress"] { display:none !important; }

/* Gradio 杂项清除 */
.gr-group, .gr-box { border:none !important; box-shadow:none !important; background:transparent !important; }
"""

# ===================== JS =====================

# 🎙️ 纯浏览器 MediaRecorder 录音 —— 完全脱离 Gradio Audio 组件
# 录音完成后将 webm base64 写入隐藏 Textbox，触发 Gradio 事件链传给后端
RECORD_JS = """
() => {
    const btn = document.getElementById('btn-r');
    if (!btn) return;

    const setStatus = (text) => {
        const el = document.getElementById('status-tip');
        if (el) el.innerHTML = '<p class="s-tip">' + text + '</p>';
    };

    // 全局录音状态
    if (!window.__xzAudio) {
        window.__xzAudio = { recording: false, mediaRecorder: null, chunks: [], stream: null };
    }
    const audio = window.__xzAudio;

    if (!btn.classList.contains('recording')) {
        // ══════════ 开始录音 ══════════
        btn.classList.add('recording');
        btn.textContent = '⏹️';
        btn.style.background = 'linear-gradient(135deg,#ef4444,#dc2626)';
        btn.style.boxShadow = '0 4px 20px rgba(239,68,68,0.80)';
        setStatus('🔴 正在录音中，再次点击停止...');

        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(stream => {
                audio.stream  = stream;
                audio.chunks  = [];

                const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                    ? 'audio/webm;codecs=opus'
                    : MediaRecorder.isTypeSupported('audio/webm')
                        ? 'audio/webm'
                        : 'audio/mp4';

                audio.mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
                audio.mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audio.chunks.push(e.data); };

                audio.mediaRecorder.onstop = () => {
                    const blob = new Blob(audio.chunks, { type: audio.mediaRecorder.mimeType });
                    console.log('✅ 录音完成:', (blob.size/1024).toFixed(1), 'KB', '类型:', blob.type);
                    setStatus('⏳ 正在识别语音...');
                    sendBlobToGradio(blob);
                };

                audio.mediaRecorder.start(200);   // 每 200 ms 触发一次 ondataavailable
                audio.recording = true;
                console.log('🎙️ 录音开始, mimeType:', mime);
            })
            .catch(err => {
                console.error('❌ 麦克风错误:', err);
                setStatus('❌ 无法访问麦克风，请检查权限');
                btn.classList.remove('recording');
                btn.textContent = '🎤';
                btn.style.background = '';
                btn.style.boxShadow = '';
            });

    } else {
        // ══════════ 停止录音 ══════════
        btn.classList.remove('recording');
        btn.textContent = '🎤';
        btn.style.background = '';
        btn.style.boxShadow = '';

        if (audio.mediaRecorder && audio.recording) {
            audio.recording = false;
            audio.mediaRecorder.stop();                                  // 触发 onstop
            audio.stream && audio.stream.getTracks().forEach(t => t.stop());
        }
    }

    // ── 将 Blob 转 base64 → 写入隐藏 Textbox → 触发 Gradio change 事件 ──
    function sendBlobToGradio(blob) {
        const reader = new FileReader();
        reader.onloadend = () => {
            // reader.result 形如 "data:audio/webm;base64,AAAA..."
            const b64 = reader.result.split(',')[1];
            if (!b64) { console.error('base64 转换失败'); return; }

            // 找到隐藏 Textbox 的 <textarea>
            const zone = document.querySelector('.voice-hidden-zone');
            if (!zone) { console.error('找不到 .voice-hidden-zone'); return; }
            const textarea = zone.querySelector('textarea');
            if (!textarea) { console.error('找不到隐藏 textarea'); return; }

            // 写值并触发 input + change（Gradio 监听的是 input 事件）
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            nativeSetter.call(textarea, b64);
            textarea.dispatchEvent(new Event('input',  { bubbles: true }));
            textarea.dispatchEvent(new Event('change', { bubbles: true }));
            console.log('✅ base64 已写入隐藏 Textbox，长度:', b64.length);
        };
        reader.readAsDataURL(blob);
    }
}
"""

RESET_MIC_JS = """
() => {
    const btn = document.getElementById('btn-r');
    if (!btn) return;
    btn.classList.remove('recording');
    btn.textContent = '🎤';
    btn.setAttribute('data-label', '录音');
    btn.style.background = '';
    btn.style.boxShadow = '';
    btn.style.animation = '';
}
"""

# 录音处理完成后：
#   1. 用 nativeSetter 把隐藏 Textbox 清空，但【不 dispatch 任何事件】，避免二次触发 change
#   2. 恢复录音按钮 UI
AFTER_VOICE_JS = """
() => {
    // ── 静默清空隐藏 Textbox（不触发 Gradio change 事件）──
    const zone = document.querySelector('.voice-hidden-zone');
    if (zone) {
        const ta = zone.querySelector('textarea');
        if (ta) {
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(ta, '');
            // 故意不 dispatch input / change，防止再次触发后端
        }
    }
    // ── 恢复录音按钮 ──
    const btn = document.getElementById('btn-r');
    if (btn) {
        btn.classList.remove('recording');
        btn.textContent = '🎤';
        btn.setAttribute('data-label', '录音');
        btn.style.background = '';
        btn.style.boxShadow = '';
        btn.style.animation = '';
    }
}
"""

CAMERA_JS = """
async () => {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('当前浏览器不支持摄像头访问');
            return;
        }
        if (window.__xzCam && window.__xzCam.wrap) {
            if (window.__xzCam.stream) {
                window.__xzCam.stream.getTracks().forEach(t => t.stop());
            }
            window.__xzCam.wrap.remove();
            window.__xzCam = null;
            return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const wrap = document.createElement('div');
        wrap.style.cssText = 'position:fixed;right:16px;bottom:16px;width:260px;height:170px;'
            + 'border-radius:10px;border:1px solid rgba(168,85,247,0.45);'
            + 'background:rgba(8,4,22,0.95);z-index:9999;overflow:hidden;cursor:move;user-select:none;';

        const video = document.createElement('video');
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;
        video.srcObject = stream;
        video.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
        wrap.appendChild(video);
        document.body.appendChild(wrap);

        let dragging = false, offsetX = 0, offsetY = 0;
        wrap.addEventListener('mousedown', e => {
            dragging = true;
            const rect = wrap.getBoundingClientRect();
            offsetX = e.clientX - rect.left;
            offsetY = e.clientY - rect.top;
            wrap.style.right = 'auto'; wrap.style.bottom = 'auto';
            wrap.style.left = rect.left + 'px'; wrap.style.top = rect.top + 'px';
        });
        window.addEventListener('mousemove', e => {
            if (!dragging) return;
            wrap.style.left = (e.clientX - offsetX) + 'px';
            wrap.style.top  = (e.clientY - offsetY) + 'px';
        });
        window.addEventListener('mouseup', () => { dragging = false; });

        window.__xzCam = { wrap, stream };
    } catch (e) {
        alert('无法打开摄像头，请检查权限设置');
    }
}
"""

# ===================== 界面 =====================
def build_app():
    with gr.Blocks(title="小智 AI 助手") as demo:
        session_id   = gr.State(lambda: f"user_{uuid.uuid4().hex[:8]}")
        chat_history = gr.State([])

        gr.HTML(
            f'<div class="app-bg" style="background-image:url(\'{BG_SRC}\');"></div>'
            if BG_SRC else
            '<div class="app-bg" style="background:linear-gradient(135deg,#0f0720,#1a0a3e);"></div>'
        )

        with gr.Row(elem_classes="app-row"):

            # ── 左侧角色区 ──────────────────────────────
            with gr.Column(elem_classes="char-col", scale=72):
                gr.HTML(
                    '<div class="status-badge"><span class="dot-live"></span>小智 AI · 在线</div>'
                    + (f'<img class="char-img" src="{CHAR_SRC}" alt="小智">'
                       if CHAR_SRC else '<div style="height:70vh;"></div>')
                )
                with gr.Row(elem_classes="ctrl-bar"):
                    btn_s = gr.Button("📂", elem_id="btn-s")
                    btn_e = gr.Button("📷", elem_id="btn-e")
                    btn_m = gr.Button("🔇", elem_id="btn-m")
                    btn_r = gr.Button("🎤", elem_id="btn-r")

            # ── 右侧聊天区 28% ──────────────────────────
            with gr.Column(elem_classes="chat-col", scale=28):
                gr.HTML('<div class="chat-hdr">💬 对话记录</div>')

                chat_display = gr.HTML(
                    value='<p class="empty-chat">✨ 开始对话吧～</p>',
                    elem_id="cb"
                )

                # 音频输出（隐藏，仅自动播放 TTS）
                with gr.Row(elem_classes="audio-zone"):
                    audio_out = gr.Audio(
                        type="filepath", autoplay=True, show_label=False,
                    )

                # 状态提示
                status_tip = gr.HTML(
                    tip("点击 🎤 开始录音，再次点击停止"),
                    elem_id="status-tip"
                )

                # ── 隐藏 Textbox：接收前端 base64 录音数据 ──
                # 完全通过 CSS 不可见；JS 直接写 textarea.value 并 dispatch input 事件
                with gr.Row(elem_classes="voice-hidden-zone"):
                    voice_b64_input = gr.Textbox(
                        value="",
                        show_label=False,
                        visible=True,        # 保持 visible=True，让 Gradio 正常渲染 DOM
                        interactive=True,
                        container=False,
                        elem_id="voice-b64-input",
                    )

                # 文字输入
                with gr.Row(elem_classes="input-zone"):
                    txt_in   = gr.Textbox(
                        placeholder="输入消息，按 Enter 发送...",
                        show_label=False, scale=8, container=False, lines=3
                    )
                    btn_send = gr.Button("▶", elem_id="btn-send", scale=0)

        # ── 事件绑定 ──────────────────────────────────
        def upd(h): return build_chat_html(h)

        async def on_text(msg, h, sid):
            if not msg or not msg.strip():
                return h, upd(h), None, gr.update(value="", interactive=True), tip("请输入内容"), gr.update(interactive=True)
            nh, ap, _ = await handle_text_chat(msg, h, sid)
            return nh, upd(nh), ap, gr.update(value="", interactive=True), tip("✅ 已回复"), gr.update(interactive=True)

        def on_text_start():
            return gr.update(interactive=True), tip("正在回答中..."), gr.update(interactive=False)

        def on_clear():
            sid = f"user_{uuid.uuid4().hex[:8]}"
            return [], '<p class="empty-chat">✨ 新会话已开始</p>', None, tip("✨ 已清空"), sid

        def open_data_folder():
            db_list = [n for n in ["crm_database.db", "crm_customers.db"] if (BASE_DIR / n).exists()]
            try:
                # 导出对话记录
                export_path = BASE_DIR / "chat_records_readable.txt"
                crm_service.export_readable_records(str(export_path))
                
                # 【新增】导出客户画像信息（类似 chat_records_readable.txt，但展示用户画像）
                from config import USER_PROFILES_PATH
                profiles_path = Path(USER_PROFILES_PATH)
                crm_service.export_user_profiles(str(profiles_path))
                
                if os.name == "nt":
                    os.startfile(str(BASE_DIR))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(BASE_DIR)])
                else:
                    subprocess.Popen(["xdg-open", str(BASE_DIR)])
                
                files_generated = ["chat_records_readable.txt", "user_profiles.txt"]
                if db_list:
                    return tip(f"📂 已打开数据文件夹，已生成 {', '.join(files_generated)}（含 {', '.join(db_list)}）")
                return tip(f"📂 已打开数据文件夹，已生成 {', '.join(files_generated)}")
            except Exception as e:
                logger.error(f"打开数据文件夹失败: {e}")
                return tip("❌ 打开数据文件夹失败")

        # ── 文字发送 ──
        send_evt = btn_send.click(
            on_text_start,
            outputs=[txt_in, status_tip, btn_send],
            queue=False,
            show_progress="hidden"
        )
        send_evt.then(
            wrap_async(on_text),
            inputs=[txt_in, chat_history, session_id],
            outputs=[chat_history, chat_display, audio_out, txt_in, status_tip, btn_send],
            show_progress="hidden"
        )

        submit_evt = txt_in.submit(
            on_text_start,
            outputs=[txt_in, status_tip, btn_send],
            queue=False,
            show_progress="hidden"
        )
        submit_evt.then(
            wrap_async(on_text),
            inputs=[txt_in, chat_history, session_id],
            outputs=[chat_history, chat_display, audio_out, txt_in, status_tip, btn_send],
            show_progress="hidden"
        )

        # ── 🎤 录音按钮：只做 JS UI 切换，实际录音逻辑在 RECORD_JS 里 ──
        btn_r.click(None, js=RECORD_JS)

        # ── 隐藏 Textbox change 事件：收到 base64 录音数据后调用后端 ──
        # 注意：outputs 里【不包含】voice_b64_input，
        # 清空工作交由 JS 完成（直接设 value 但不 dispatch 事件），避免二次触发导致 audio_out 被 None 覆盖
        voice_b64_input.change(
            receive_voice_b64,
            inputs=[voice_b64_input, chat_history, session_id],
            outputs=[chat_history, chat_display, audio_out, status_tip],
            show_progress="hidden",
        ).then(None, js=AFTER_VOICE_JS)

        # ── 📂 打开数据文件夹 ──
        btn_s.click(open_data_folder, outputs=[status_tip])

        # ── 📷 摄像头 ──
        btn_e.click(None, js=CAMERA_JS)

        # ── 🔇 停止播放 ──
        btn_m.click(
            lambda h: (h, upd(h), None, tip("🔇 已停止播放")),
            inputs=[chat_history],
            outputs=[chat_history, chat_display, audio_out, status_tip]
        )

    return demo


if __name__ == "__main__":
    try:
        from config import check_config
        check_config()
    except Exception:
        pass
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=False,
        css=CSS,
        allowed_paths=[str(BASE_DIR)],
    )