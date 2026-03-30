"""
services/tts_service.py - 语音合成服务（TTS）
=============================================
TTS = Text to Speech，文字转语音
功能：把文字转换成 MP3 音频字节

使用的技术：edge-tts
  - 微软 Edge 浏览器的 TTS 服务
  - 完全免费，中文效果非常好
  - 需要联网（调用微软服务器）
  - 支持多种中文声音和情感

工作流程：
  文字 → 调用微软 TTS API → 返回 MP3 音频字节
"""

import io
import logging
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class TTSService:
    """语音合成服务类"""
    
    def __init__(self):
        """初始化 TTS 服务"""
        # 检查 edge-tts 是否安装
        try:
            import edge_tts
            logger.info("✅ TTS 服务初始化成功（edge-tts）")
        except ImportError:
            raise ImportError("edge-tts 未安装！请运行：pip install edge-tts")
    
    async def synthesize(self, text: str) -> bytes:
        """
        把文字转换成语音（MP3 格式的字节数据）
        
        参数：
            text：要合成的文字
        
        返回：
            MP3 音频数据（字节）
        """
        import edge_tts
        from config import TTS_VOICE, TTS_RATE, TTS_PITCH
        
        if not text or not text.strip():
            logger.warning("⚠️  TTS 输入文字为空，跳过合成")
            return b""
        
        # 清理文字：去掉特殊字符，避免 TTS 读出奇怪的东西
        clean_text = self._clean_text(text)
        
        logger.info(f"🔊 开始合成语音：{clean_text[:30]}...")
        
        try:
            # 创建 edge-tts 通信对象
            # voice：声音名称
            # rate：语速
            # pitch：音调
            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=TTS_VOICE,
                rate=TTS_RATE,
                pitch=TTS_PITCH
            )
            
            # 收集所有音频数据块
            # edge-tts 是流式返回的，我们把所有块合并成一个完整的 MP3
            audio_chunks = []
            
            # async for 是异步循环，遍历流式返回的数据
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # chunk["data"] 是这块音频的字节数据
                    audio_chunks.append(chunk["data"])
            
            if not audio_chunks:
                logger.warning("⚠️  TTS 没有返回音频数据")
                return b""
            
            # 把所有块合并 b"".join([...]) 是字节拼接
            audio_bytes = b"".join(audio_chunks)
            
            logger.info(f"✅ 语音合成成功，大小：{len(audio_bytes)} 字节")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"❌ 语音合成失败：{e}")
            # 如果失败，返回空字节（前端不播放声音但会显示文字）
            return b""
    
    def _clean_text(self, text: str) -> str:
        """
        清理文字，去掉 TTS 不需要读的内容
        比如颜文字、特殊标点等
        """
        import re
        
        # 去掉常见的颜文字和特殊字符
        # 但保留中文标点，因为它们影响语调
        text = re.sub(r'[（）\(\)\[\]\{\}]', '', text)  # 去掉括号
        text = re.sub(r'[*_~`]', '', text)  # 去掉 Markdown 格式
        
        # 把多个换行替换成一个
        text = re.sub(r'\n+', '，', text)
        
        return text.strip()
    
    async def get_available_voices(self) -> list:
        """
        获取所有可用的中文语音列表
        可以运行这个函数看看有哪些声音可以选
        """
        import edge_tts
        
        voices = await edge_tts.list_voices()
        # 过滤出中文语音
        chinese_voices = [v for v in voices if v["Locale"].startswith("zh-")]
        return chinese_voices


# 全局单例
_tts_instance = None

def get_tts_service() -> TTSService:
    """获取 TTS 服务实例"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSService()
    return _tts_instance


# ============================================================
# 测试代码：直接运行这个文件可以测试 TTS
# python services/tts_service.py
# ============================================================
if __name__ == "__main__":
    async def test():
        service = TTSService()
        print("正在合成语音...")
        audio = await service.synthesize("你好呀！我是小智，很高兴认识你～")
        if audio:
            # 保存为 MP3 文件
            with open("test_output.mp3", "wb") as f:
                f.write(audio)
            print(f"✅ 语音合成成功！保存为 test_output.mp3（{len(audio)} 字节）")
        else:
            print("❌ 语音合成失败")
    
    asyncio.run(test())
