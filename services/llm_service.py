"""
services/llm_service.py - 大语言模型服务（LLM）
================================================
LLM = Large Language Model，大语言模型
功能：接收用户文字，返回 AI 的回复

【改进说明】
  原版问题：只用了一个通用 system prompt，AI 完全不知道 CRM 业务知识
  改进方案：
    1. 加载 crm_knowledge.txt 知识库，注入到 system prompt
    2. 接收 CRM 上下文（当前用户信息、搜索到的客户数据）
    3. 意图识别：判断用户在问课程/客户/闲聊，动态调整回答策略
    4. 关键词匹配：从用户消息中提取人名/电话，自动查询 CRM

使用的技术：DeepSeek API（兼容 OpenAI 格式）
"""

import logging
import os
from typing import List, Dict, Any, Optional
from config import DEEPSEEK_MODEL 
logger = logging.getLogger(__name__)


class LLMService:
    """大语言模型服务类（增强版：带 CRM 知识库和意图识别）"""
    
    def __init__(self):
        """初始化：创建 OpenAI 客户端（配置为 DeepSeek）"""
        from openai import AsyncOpenAI
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        
        if not DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API Key 未配置！请设置环境变量 DEEPSEEK_API_KEY")
        
        # 创建 AsyncOpenAI 客户端
        self.client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        # 加载 CRM 知识库
        self.knowledge_base = self._load_knowledge_base()
        
        logger.info("✅ LLM 服务初始化成功（DeepSeek + CRM 知识库）")
    
    def _load_knowledge_base(self) -> str:
        """
        加载 CRM 知识库文件
        知识库包含：课程信息、产品信息、FAQ、公司信息、促销活动等
        这些内容会被注入到 system prompt 中，让 AI "知道"业务知识
        """
        from config import KNOWLEDGE_BASE_PATH
        
        try:
            if os.path.exists(KNOWLEDGE_BASE_PATH):
                with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
                    content = f.read()
                logger.info(f"📚 知识库加载成功：{len(content)} 字符")
                return content
            else:
                logger.warning(f"⚠️ 知识库文件不存在：{KNOWLEDGE_BASE_PATH}")
                return ""
        except Exception as e:
            logger.error(f"❌ 知识库加载失败：{e}")
            return ""
    
    def _build_system_prompt(
        self,
        user_info_text: str = "",
        crm_search_result: str = "",
        all_users_text: str = ""
    ) -> str:
        """
        动态构建 system prompt
        把基础人设 + 知识库 + 当前用户信息 + CRM 查询结果 拼接在一起
        
        参数：
            user_info_text：当前对话用户的 CRM 信息
            crm_search_result：搜索到的其他客户信息
            all_users_text：所有客户的摘要（用户问"有哪些客户"时使用）
        
        返回：完整的 system prompt
        """
        from config import SYSTEM_PROMPT
        
        parts = [SYSTEM_PROMPT]
        
        # 注入 CRM 知识库
        if self.knowledge_base:
            parts.append(f"\n\n## 【CRM 知识库】\n以下是公司的业务知识，回答用户问题时请参考这些信息：\n\n{self.knowledge_base}")
        
        # 注入当前用户信息（让 AI 知道在和谁对话）
        if user_info_text:
            parts.append(f"\n\n## 【当前对话用户的信息】\n{user_info_text}\n\n请在回答时参考这些信息，比如称呼用户的名字。")
        
        # 注入 CRM 客户搜索结果（用户提到某个客户时）
        if crm_search_result:
            parts.append(f"\n\n## 【CRM 客户数据】\n以下是 CRM 系统中查询到的客户信息：\n{crm_search_result}")
        
        # 注入所有客户列表（用户问"有哪些客户"时）
        if all_users_text:
            parts.append(f"\n\n## 【CRM 全部客户列表】\n{all_users_text}")
        
        return "\n".join(parts)
    
    def _extract_search_keywords(self, user_message: str) -> List[str]:
        """
        从用户消息中提取可能的人名/电话/公司名等关键词
        用于在 CRM 中搜索客户
        
        这是一个简单的启发式方法：
        - 提取中文人名（2-4个汉字的组合）
        - 提取手机号（11位数字）
        - 提取带引号的内容
        
        参数：
            user_message：用户输入的消息
            
        返回：可能的关键词列表
        """
        import re
        keywords = []
        
        # 提取手机号
        phone_pattern = r'1[3-9]\d{9}'
        phones = re.findall(phone_pattern, user_message)
        keywords.extend(phones)
        
        # 提取引号中的内容（如"张三"、'李四'）
        quoted = re.findall(r'[""「」『』](.+?)[""「」『』]', user_message)
        keywords.extend(quoted)
        
        # 提取"叫xxx"、"名叫xxx"、"找xxx"、"查xxx"、"问xxx的客户"中的人名
        name_patterns = [
            r'(?:叫|名叫|是|找|查|查询|搜索|问|有|关于)\s*[""「]?([\u4e00-\u9fa5]{2,4})[""」]?',
            r'(?:客户|用户|人)\s*[""「]?([\u4e00-\u9fa5]{2,4})[""」]?',
        ]
        for pattern in name_patterns:
            names = re.findall(pattern, user_message)
            keywords.extend(names)
        
        return keywords
    
    async def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        crm_context: Optional[Dict[str, str]] = None
    ) -> str:
        """
        发送消息给 AI，获取回复（增强版）
        
        【核心改进】
        原版：只用通用 system prompt，AI 不知道任何业务知识
        改进：
          1. 动态构建 system prompt，注入知识库 + CRM 数据
          2. 从用户消息中提取关键词，自动搜索 CRM
          3. 根据意图智能选择回答策略
        
        参数：
            user_message：用户输入的文字
            conversation_history：之前的对话历史（可选）
            crm_context：CRM 上下文信息（可选），包含：
                - user_info_text：当前用户信息
                - crm_search_result：搜索到的客户信息
                - all_users_text：所有客户列表
        
        返回：AI 的回复文字
        """
        from config import DEEPSEEK_MODEL, LLM_MAX_TOKENS
        
        # 解析 CRM 上下文
        crm_ctx = crm_context or {}
        user_info_text = crm_ctx.get("user_info_text", "")
        crm_search_result = crm_ctx.get("crm_search_result", "")
        all_users_text = crm_ctx.get("all_users_text", "")
        
        # 动态构建 system prompt（包含知识库 + CRM 数据）
        system_prompt = self._build_system_prompt(
            user_info_text=user_info_text,
            crm_search_result=crm_search_result,
            all_users_text=all_users_text
        )
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 加入历史对话（最多保留最近10轮）
        if conversation_history:
            recent_history = conversation_history[-20:]
            messages.extend(recent_history)
        
        # 加入当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        try:
            logger.info(f"💬 发送给 AI：{user_message[:50]}...")
            
            # 调用 DeepSeek API
            response = await self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0.7,
                stream=False
            )
            
            ai_reply = response.choices[0].message.content
            
            if not ai_reply:
                ai_reply = "哎呀，我好像没想好怎么回答呢～"
            
            logger.info(f"🤖 AI 回复：{ai_reply[:50]}...")
            return ai_reply
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ LLM 调用失败：{error_msg}")
            
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
