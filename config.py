"""
config.py - 项目配置文件
==========================
所有的配置项都在这里，修改配置只需要改这一个文件。

什么是配置文件？
  就像餐厅的菜单设置，你不需要改厨房的做法（代码），
  只需要改菜单（配置）就能改变行为。
"""

import os

# ============================================================
# DeepSeek 大模型配置
# ============================================================

# API 密钥：从环境变量读取（你已经配置好了）
# 如果环境变量没有，会用这里的默认值（留空则报错）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# DeepSeek API 地址（固定的，不用改）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 使用的模型名称
# deepseek-chat：通用对话模型（推荐）
# deepseek-reasoner：推理增强模型（更贵但更聪明）
DEEPSEEK_MODEL = "deepseek-chat"

# 每次回复的最大字数（token 数，中文约等于字数）
LLM_MAX_TOKENS = 500

# AI 助手的"人设"提示词（System Prompt）
# 这段话告诉 AI 它是谁、怎么说话
SYSTEM_PROMPT = """你是一位温柔可爱的 AI 助手，名叫小智。
你的特点：
- 说话温柔亲切，喜欢用"呢"、"哦"、"～"等语气词
- 回答简洁，每次不超过100个字
- 遇到不知道的问题，诚实说"不太清楚呢"
- 偶尔用颜文字表情，比如(≧▽≦)

请用简体中文回答所有问题。"""

# ============================================================
# 语音识别（ASR）配置
# ============================================================

# Whisper 模型大小，影响准确率和速度
# tiny  → 最快，约39MB，准确率一般
# base  → 推荐，约74MB，速度和准确率平衡
# small → 较慢，约244MB，准确率更好
# medium→ 很慢，约769MB，准确率很好（GPU 推荐）
WHISPER_MODEL_SIZE = "base"

# 运行设备：cpu 或 cuda（如果你有 NVIDIA 显卡可以改成 cuda）
WHISPER_DEVICE = "cpu"

# 计算精度：cpu 用 int8，cuda 可以用 float16
WHISPER_COMPUTE_TYPE = "int8"

# 语言设置（zh 表示中文，None 表示自动检测）
WHISPER_LANGUAGE = "zh"

# ============================================================
# 语音合成（TTS）配置
# ============================================================

# 语音角色（微软 Edge TTS 的声音）
# 中文声音推荐：
# zh-CN-XiaoxiaoNeural  → 晓晓（温柔女声，推荐！）
# zh-CN-YunxiNeural     → 云希（年轻男声）
# zh-CN-XiaoyiNeural    → 晓伊（活泼女声）
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# 语速（-50% 到 +100%，0 是正常速度）
TTS_RATE = "+10%"

# 音调（-50Hz 到 +50Hz，0 是正常）
TTS_PITCH = "+0Hz"

# ============================================================
# 服务器配置
# ============================================================

# 后端服务地址和端口
SERVER_HOST = "0.0.0.0"   # 0.0.0.0 表示接受所有来源的连接
SERVER_PORT = 8000

# ============================================================
# 数据库配置
# ============================================================

# SQLite 数据库文件路径（会自动创建）
DATABASE_URL = "sqlite:///./crm_database.db"

# ============================================================
# 检查配置是否有问题
# ============================================================

def check_config():
    """启动时检查关键配置"""
    errors = []
    
    if not DEEPSEEK_API_KEY:
        errors.append(
            "❌ 缺少 DEEPSEEK_API_KEY！\n"
            "   解决方法：\n"
            "   Windows: set DEEPSEEK_API_KEY=sk-你的密钥\n"
            "   Mac/Linux: export DEEPSEEK_API_KEY=sk-你的密钥"
        )
    
    if errors:
        print("\n" + "="*50)
        print("⚠️  配置检查失败，请解决以下问题后重新启动：")
        print("="*50)
        for err in errors:
            print(err)
        print("="*50 + "\n")
        return False
    
    print("✅ 配置检查通过")
    return True
