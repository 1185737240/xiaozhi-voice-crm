"""
services/asr_service.py - 语音识别服务（ASR）
=============================================
ASR = Automatic Speech Recognition，自动语音识别
功能：把录音文件（音频字节）转换成文字

使用的技术：faster-whisper
  - 这是 OpenAI Whisper 模型的优化版本
  - 支持中文，离线运行（不需要联网）
  - 第一次运行会自动下载模型到 ~/.cache/huggingface/

工作流程：
  音频字节 → 保存为临时WAV文件 → Whisper识别 → 返回文字
"""

import os
import io
import tempfile
import logging
import numpy as np
import soundfile as sf

# 配置日志（用来在控制台打印信息，方便调试）
# logging.INFO 表示打印信息级别的日志
logger = logging.getLogger(__name__)


class ASRService:
    """
    语音识别服务类
    
    什么是"类"？
      可以把它理解为一个"工具箱"，
      里面有初始化工具的方法（__init__）
      和使用工具的方法（transcribe）。
    """
    
    def __init__(self):
        """
        初始化方法：当我们创建 ASRService 对象时自动调用
        主要工作：加载 Whisper 模型（只加载一次，节省时间）
        """
        self.model = None  # 先设为 None，等需要时再加载
        self._load_model()
    
    def _load_model(self):
        """
        加载 Whisper 模型
        下划线开头的方法是"私有方法"，表示只在内部使用
        """
        try:
            # 在这里才 import，因为加载模型需要时间
            # 如果放在文件顶部，每次启动都会等待
            from faster_whisper import WhisperModel
            
            # 从配置文件导入设置
            from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
            
            logger.info(f"⏳ 正在加载 Whisper 模型（{WHISPER_MODEL_SIZE}）...")
            logger.info("   第一次加载会下载模型文件，请耐心等待...")
            
            # 创建 WhisperModel 对象
            # model_size_or_path：模型大小，"base" 约74MB
            # device：运行设备，"cpu" 或 "cuda"
            # compute_type：计算精度，cpu 用 "int8" 最快
            self.model = WhisperModel(
                model_size_or_path=WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            
            logger.info("✅ Whisper 模型加载成功！")
            
        except ImportError:
            # 如果 faster-whisper 没有安装，给出提示
            logger.error("❌ faster-whisper 未安装，请运行：pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"❌ 模型加载失败：{e}")
            raise
    
    async def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        """
        把音频字节转换成文字
        
        参数：
            audio_bytes：音频数据（字节格式）
            audio_format：音频格式，默认 "wav"，也支持 "webm"、"mp4"
        
        返回：
            识别出来的文字字符串
        """
        from config import WHISPER_LANGUAGE, ASR_INITIAL_PROMPT
        
        if not self.model:
            raise RuntimeError("Whisper 模型未加载！")
        
        if not audio_bytes:
            return ""
        
        # 使用临时文件处理音频
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(audio_bytes)
        
        try:
            logger.info(f"🎤 开始识别音频，格式：{audio_format}，大小：{len(audio_bytes)} 字节")
            
            # 调用 Whisper 进行语音识别
            segments, info = self.model.transcribe(
                tmp_path,
                language=WHISPER_LANGUAGE,
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 1000,   # 静音超过1秒才切断（原来500ms太敏感）
                    "min_speech_duration_ms": 250,     # 语音片段最短250ms（过滤掉太短的噪声）
                    "speech_pad_ms": 200               # 语音前后各留200ms缓冲
                },
                initial_prompt=ASR_INITIAL_PROMPT      # 引导模型输出简体中文
            )
            
            text_parts = [segment.text for segment in segments]
            result = "".join(text_parts).strip()
            
            if result:
                logger.info(f"✅ 识别结果：{result}")
            else:
                logger.info("⚠️  没有识别到任何内容（可能是静音）")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 语音识别失败：{e}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    async def transcribe_from_browser(self, audio_bytes: bytes, mime_type: str = "webm") -> str:
        """
        处理浏览器录音（支持 WebM 和 MP4 两种格式）
        
        浏览器的 MediaRecorder 在不同系统上可能录制不同格式：
        - Chrome/Edge（Windows）：通常是 WebM/Opus
        - Chrome/Edge（Mac/Safari）：可能是 MP4/AAC
        - Firefox：通常是 WebM/Opus
        
        参数：
            audio_bytes：音频字节数据
            mime_type：MIME 类型，"webm" 或 "mp4"
        """
        try:
            from pydub import AudioSegment
            
            # 根据实际格式进行转换（不再硬编码 webm）
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=mime_type)
            
            # 统一转为 16kHz 单声道 WAV（Whisper 最友好的格式）
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()
            
            logger.info(f"🔄 音频转换成功：{mime_type} → WAV（16kHz 单声道）")
            return await self.transcribe(wav_bytes, "wav")
            
        except Exception as e:
            logger.error(f"音频转换失败（{mime_type}）：{e}，尝试直接识别...")
            # 如果转换失败，直接尝试原始数据
            return await self.transcribe(audio_bytes, mime_type)


# 创建全局单例（整个程序只用一个 ASR 实例，节省内存）
# 单例模式：保证一个类只有一个实例
_asr_instance = None

def get_asr_service() -> ASRService:
    """
    获取 ASR 服务实例
    使用"懒加载"：第一次调用时才创建，之后复用同一个
    """
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASRService()
    return _asr_instance
