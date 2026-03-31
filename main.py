"""
main.py — FastAPI 主服务
暴露 /ask 接口，支持文本和语音两种输入，返回文本+语音两种输出
"""
import base64
import logging
import tempfile
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 服务实例（启动时加载，全局复用）────────────────────────
from services.asr_service import get_asr_service
from services.llm_service import get_llm_service
from services.tts_service import get_tts_service
from services.crm_service import get_crm_service

asr = tts = llm = crm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化所有服务（只加载一次，节省内存）"""
    global asr, tts, llm, crm
    logger.info("⏳ 正在初始化服务...")
    asr = get_asr_service()
    llm = get_llm_service()
    tts = get_tts_service()
    crm = get_crm_service()
    logger.info("✅ 所有服务初始化完成，API 已就绪")
    yield
    logger.info("🛑 服务已关闭")


app = FastAPI(
    title="小智语音问答 API",
    description="""
## 轻量级语音问答服务

参考 xiaozhi-esp32-server 项目，提取核心语音交互逻辑，构建独立 REST API。

### 功能
- **文本问答**：发送文字，返回 AI 文字回复 + 语音回复
- **语音问答**：上传录音文件，自动识别后返回文字 + 语音回复
- **CRM 集成**：每次对话自动提取用户信息（姓名/电话/邮箱等）存入数据库

### 接口入口
- `POST /ask` — 统一问答接口（文本或语音二选一）
- `GET /health` — 服务健康检查
- `GET /crm/users` — 查询所有 CRM 用户
    """,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 数据模型 ──────────────────────────────────────────────

class AskResponse(BaseModel):
    """统一响应格式"""
    session_id: str
    input_text: str            # 用户实际输入（语音模式下是 ASR 识别结果）
    reply_text: str            # AI 文字回复
    reply_audio_base64: str    # AI 语音回复（MP3，base64 编码）
    crm_info: dict             # 本次对话提取到的用户信息


# ── 核心接口 ──────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="统一问答接口",
    description="""
支持两种输入模式（二选一）：

**模式 A — 文本输入（application/json）**
```json
{
  "text": "你好，我叫张三，电话13800138000",
  "session_id": "user_001",
  "return_audio": true
}
```

**模式 B — 语音输入（multipart/form-data）**
```
audio=<录音文件>
session_id=user_001
return_audio=true
```

两种模式返回格式相同。
    """
)
async def ask(
    # 文本输入字段（JSON 或 form-data 均可）
    text: Optional[str] = Form(None, description="文字输入（与 audio 二选一）"),
    session_id: str = Form("default", description="用户会话 ID，用于区分不同用户"),
    return_audio: bool = Form(True, description="是否返回语音回复（base64 MP3）"),
    # 语音输入字段
    audio: Optional[UploadFile] = File(None, description="录音文件（与 text 二选一，支持 WAV/WebM/MP3）"),
):
    # ① 校验：text 和 audio 必须有一个
    if not text and not audio:
        raise HTTPException(status_code=400, detail="请提供 text（文字）或 audio（录音文件），二选一")

    # ② 语音模式：先 ASR 识别成文字
    if audio:
        audio_bytes = await audio.read()
        suffix = os.path.splitext(audio.filename or "")[-1].lower() or ".webm"
        # WebM 特殊处理
        if suffix in (".webm", ".ogg"):
            input_text = await asr.transcribe_from_webm(audio_bytes)
        else:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name
            try:
                input_text = await asr.transcribe(audio_bytes, suffix.lstrip("."))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if not input_text:
            raise HTTPException(status_code=422, detail="语音识别失败，未能识别到有效内容，请重试")
    else:
        input_text = text.strip()

    # ③ 获取历史对话（构建多轮上下文）
    history_msgs = crm.get_session_messages(session_id)
    llm_history = [
        {"role": m["role"], "content": m["content"]}
        for m in history_msgs[-20:]  # 最近 10 轮
    ]

    # ④ 调用 LLM 获取回复
    reply_text = await llm.chat(input_text, llm_history)

    # ⑤ 保存本轮对话到数据库
    crm.save_message(session_id, "user", input_text)
    crm.save_message(session_id, "assistant", reply_text)

    # ⑥ LLM 提取用户信息（CRM）
    crm_info = {}
    try:
        conv_text = crm.get_session_text(session_id)
        crm_info = await llm.extract_user_info(conv_text)
        if crm_info:
            crm.update_user_info(session_id, crm_info)
    except Exception as e:
        logger.warning(f"CRM 提取失败（不影响主流程）: {e}")

    # ⑦ TTS 合成语音
    audio_b64 = ""
    if return_audio:
        import re
        tts_text = re.sub(r"\([^)]*\)", " ", reply_text)
        tts_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？、]", " ", tts_text).strip()
        audio_bytes = await tts.synthesize(tts_text or reply_text)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode()

    return AskResponse(
        session_id=session_id,
        input_text=input_text,
        reply_text=reply_text,
        reply_audio_base64=audio_b64,
        crm_info=crm_info,
    )


# ── 辅助接口 ──────────────────────────────────────────────

@app.get("/health", summary="健康检查", tags=["系统"])
async def health():
    return {"status": "ok", "message": "小智语音问答服务运行正常"}


@app.get("/crm/users", summary="查询所有 CRM 用户", tags=["CRM"])
async def get_crm_users():
    """返回所有已存储的用户信息列表"""
    users = crm.get_all_users()
    return {"count": len(users), "users": users}


@app.get("/crm/users/{session_id}", summary="查询单个用户", tags=["CRM"])
async def get_crm_user(session_id: str):
    """根据 session_id 查询单个用户信息"""
    user = crm.get_user(session_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


# ── 启动入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import SERVER_HOST, SERVER_PORT, check_config
    check_config()
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, reload=False)