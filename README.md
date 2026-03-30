# 小智语音问答服务

> 参考 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 项目，提取核心语音交互逻辑，构建轻量级独立 REST API 服务，脱离 ESP32 硬件独立运行。

## 系统结构图

```
┌───────────────────────────────────────────────────────────┐
│              客户端（Postman / curl / 浏览器）              │
└────────────────────┬──────────────────────────────────────┘
                     │ HTTP 请求
                     ▼
┌───────────────────────────────────────────────────────────┐
│              FastAPI 主服务  main.py  port:8000            │
│                                                            │
│   POST /ask          GET /health      GET /crm/users       │
│   文本或语音输入      健康检查          查询用户列表          │
└──────┬──────────────────────────────────────────┬─────────┘
       │                                          │
       ▼                                          ▼
┌─────────────────────────────┐    ┌─────────────────────────┐
│        services/            │    │      crm_service.py      │
│                             │    │                          │
│  asr_service.py             │    │  SQLite 数据库           │
│  faster-whisper 离线识别    │    │  自动提取用户信息         │
│  WebM/WAV → 文字            │    │  姓名/电话/邮箱/需求      │
│                             │    └─────────────────────────┘
│  llm_service.py             │
│  DeepSeek API               │
│  多轮对话 + 信息提取         │
│                             │
│  tts_service.py             │
│  edge-tts 微软语音合成       │
│  文字 → MP3 音频字节         │
└─────────────────────────────┘
```

## 文件结构

```
xiaozhi-voice-crm/
├── main.py                  # FastAPI 主程序（REST API 入口）
├── gradio_app.py            # Gradio 可视化界面（可选启动）
├── config.py                # 所有配置项（API Key、模型、端口等）
├── requirements.txt         # Python 依赖列表
├── README.md                # 本文档
├── postman_collection.json  # Postman 接口测试集合
└── services/
    ├── asr_service.py       # 语音识别（faster-whisper）
    ├── llm_service.py       # 大语言模型（DeepSeek）
    ├── tts_service.py       # 语音合成（edge-tts）
    └── crm_service.py       # 用户信息管理（SQLite）
```

## 快速部署

### 1. 环境要求

- Python 3.10+
- DeepSeek API Key（[申请地址](https://platform.deepseek.com)）

### 2. 安装依赖

```bash
# 创建 conda 虚拟环境
conda create -n xiaozhi python=3.10 -y
conda activate xiaozhi

# Windows 需要安装 ffmpeg（音频格式转换）
conda install ffmpeg -y

# 安装 Python 依赖（清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置 API Key

```bash
# Windows
set DEEPSEEK_API_KEY=sk-你的密钥

# macOS / Linux
export DEEPSEEK_API_KEY=sk-你的密钥
```

### 4. 启动 FastAPI 服务

```bash
python main.py
```

启动成功输出：
```
✅ 配置检查通过
✅ Whisper 模型加载成功！
✅ 所有服务初始化完成，API 已就绪
INFO: Uvicorn running on http://0.0.0.0:8000
```

在线接口文档：打开浏览器访问 `http://127.0.0.1:8000/docs`

### 5. 启动 Gradio 界面（可选）

```bash
python gradio_app.py
```

浏览器打开 `http://127.0.0.1:7860`

---

## API 接口文档

### POST /ask — 统一问答接口

支持文本和语音两种输入方式（二选一）。

**文本输入示例：**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -F "text=你好，我叫张三，手机13800138000" \
  -F "session_id=user_001" \
  -F "return_audio=true"
```

**语音输入示例：**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -F "audio=@recording.wav" \
  -F "session_id=user_001" \
  -F "return_audio=true"
```

**响应格式：**

```json
{
  "session_id": "user_001",
  "input_text": "你好，我叫张三，手机13800138000",
  "reply_text": "你好张三～有什么我可以帮你的呢？(≧▽≦)",
  "reply_audio_base64": "<MP3音频的base64字符串>",
  "crm_info": {
    "name": "张三",
    "phone": "13800138000"
  }
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 二选一 | 文字输入内容 |
| audio | file | 二选一 | 录音文件（WAV/WebM/MP3） |
| session_id | string | 否 | 用户会话ID，用于多轮对话，默认 "default" |
| return_audio | bool | 否 | 是否返回语音回复，默认 true |

---

### GET /health — 健康检查

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "message": "小智语音问答服务运行正常"}
```

---

### GET /crm/users — 查询所有用户

```bash
curl http://127.0.0.1:8000/crm/users
```

```json
{
  "count": 2,
  "users": [
    {
      "session_id": "user_001",
      "name": "张三",
      "phone": "13800138000",
      "email": "",
      "needs": "了解课程价格"
    }
  ]
}
```

---

## 关于"暴露 API 接口"

**暴露 API 接口**的意思是：你的服务启动后，其他程序（或人）只要知道你的服务器地址，就可以通过 HTTP 请求调用你的功能。

- **本地开发时**：地址是 `http://127.0.0.1:8000`，只有你自己的电脑能访问
- **部署到服务器后**：地址变成 `http://你的服务器IP:8000`，任何人都可以通过这个地址调用你的 `/ask` 接口，发送文字或语音，得到 AI 回复

这就是为什么叫"独立部署"——服务和调用方解耦，任何客户端（手机 App、网页、ESP32 设备）只需要发 HTTP 请求就能使用。

---

## Postman 使用

1. 打开 Postman → Import → 选择 `postman_collection.json`
2. 设置集合变量 `base_url` 为 `http://127.0.0.1:8000`
3. 依次测试各接口，点击 Send 查看响应

---

## 常见问题

| 问题 | 解决方法 |
|------|---------|
| `DEEPSEEK_API_KEY 未配置` | 按第3步设置环境变量 |
| `ffmpeg not found` | `conda install ffmpeg -y` |
| 端口 8000 被占用 | 修改 `config.py` 中 `SERVER_PORT` |
| Whisper 下载模型慢 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
