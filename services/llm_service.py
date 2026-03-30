"""
services/llm_service.py - 大语言模型服务（LLM）
================================================
LLM = Large Language Model，大语言模型
功能：接收用户文字，返回 AI 的回复

使用的技术：DeepSeek API（兼容 OpenAI 格式）
  - DeepSeek 的 API 和 OpenAI 的 API 格式完全一样
  - 只需要换 base_url 和 api_key 就能用

什么是"对话历史"？
  就像你跟人聊天，AI 也需要记住之前说了什么，
  才能理解"它"指的是什么。这叫"上下文"。
"""

import logging
from typing import List, Dict, Any, Optional
from config import DEEPSEEK_MODEL 
logger = logging.getLogger(__name__)


class LLMService:
    """大语言模型服务类"""
    
    def __init__(self):
        """初始化：创建 OpenAI 客户端（配置为 DeepSeek）"""
        from openai import AsyncOpenAI
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API Key 未配置！请设置环境变量 DEEPSEEK_API_KEY")
        
        # 创建 AsyncOpenAI 客户端
        # AsyncOpenAI：异步客户端（不阻塞程序运行）
        # base_url：把请求发到 DeepSeek 而不是 OpenAI
        # api_key：你的 DeepSeek 密钥
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        logger.info("✅ LLM 服务初始化成功（DeepSeek）")
    
    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        发送消息给 AI，获取回复
        
        参数：
            user_message：用户输入的文字
            conversation_history：之前的对话历史（可选）
                格式：[
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好呀～"}
                ]
        
        返回：AI 的回复文字
        """
        from config import DEEPSEEK_MODEL, LLM_MAX_TOKENS, SYSTEM_PROMPT
        
        # 构建消息列表
        # 第一条是系统提示（告诉 AI 它的人设）
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # 加入历史对话（最多保留最近10轮，避免超出 token 限制）
        if conversation_history:
            # 取最后 20 条（10轮对话，每轮2条）
            recent_history = conversation_history[-20:]
            messages.extend(recent_history)
        
        # 加入当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        try:
            logger.info(f"💬 发送给 AI：{user_message[:50]}...")  # 只打印前50字
            
            # 调用 DeepSeek API
            # await 表示等待这个异步操作完成
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,    # 创造性：0 最死板，1 最随机，0.7 是平衡值
                stream=False        # 不使用流式传输（一次返回完整结果）
            )
            
            # 从响应中提取文字
            # response.choices[0].message.content 是 AI 的回复
            ai_reply = response.choices[0].message.content
            
            if not ai_reply:
                ai_reply = "哎呀，我好像没想好怎么回答呢～"
            
            logger.info(f"🤖 AI 回复：{ai_reply[:50]}...")
            return ai_reply
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ LLM 调用失败：{error_msg}")
            
            # 根据错误类型给出友好提示
            if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                return "哎呀，我的密钥好像有问题呢，请检查 DEEPSEEK_API_KEY 是否配置正确～"
            elif "rate_limit" in error_msg.lower():
                return "太多人问我了，我需要休息一下，稍后再试试吧～"
            else:
                return f"出了一点小问题呢：{error_msg[:100]}"
    
    async def extract_user_info(self, conversation_text: str) -> Dict[str, Any]:
        """
        从对话文本中提取用户个人信息
        用于 CRM 系统
        
        参数：
            conversation_text：一段对话内容
        
        返回：提取到的信息字典，例如：
            {
                "name": "张三",
                "phone": "13812345678",
                "email": "zhangsan@qq.com",
                "address": "北京市朝阳区",
                "needs": "想了解产品价格"
            }
        """
        prompt = f"""请从以下对话中提取用户的个人信息。
只提取用户明确说出的信息，不要猜测。
如果某项信息没有提到，对应字段留空字符串。

对话内容：
{conversation_text}

请严格按照以下 JSON 格式返回（不要有多余文字）：
{{
    "name": "姓名",
    "phone": "手机号",
    "email": "邮箱",
    "address": "地址",
    "company": "公司",
    "needs": "需求或意向",
    "other": "其他重要信息"
}}"""
        
        try:
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1  # 低温度：更严格按格式输出
            )
            
            import json
            result_text = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON
            # 有时 AI 会加上 ```json ... ``` 的代码块标记，需要去掉
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            info = json.loads(result_text)
            
            # 过滤掉空值，只保留有内容的字段
            info = {k: v for k, v in info.items() if v and v.strip()}
            
            return info
            
        except Exception as e:
            logger.error(f"信息提取失败：{e}")
            return {}


# 全局单例
_llm_instance = None

def get_llm_service() -> LLMService:
    """获取 LLM 服务实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMService()
    return _llm_instance
