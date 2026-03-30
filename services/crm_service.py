"""
services/crm_service.py - CRM 用户信息管理服务
================================================
CRM = Customer Relationship Management，客户关系管理
功能：
  1. 从对话中自动提取用户信息（姓名、电话等）
  2. 把信息存入 SQLite 数据库
  3. 提供查询接口

数据库结构：
  users 表：存储用户基本信息
    - id：自动编号
    - session_id：对话 ID（用来区分不同用户）
    - name：姓名
    - phone：手机号
    - email：邮箱
    - address：地址
    - company：公司
    - needs：需求
    - other：其他信息
    - created_at：创建时间
    - updated_at：更新时间

  messages 表：存储对话记录
    - id：自动编号
    - session_id：对话 ID
    - role：角色（user 或 assistant）
    - content：消息内容
    - timestamp：时间戳
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

# 创建数据库基础类
# declarative_base() 是 SQLAlchemy 的基础，所有数据库表都继承它
Base = declarative_base()


# ============================================================
# 数据库模型（相当于定义表结构）
# ============================================================

class User(Base):
    """
    用户信息表
    每个字段对应数据库里的一列
    """
    __tablename__ = "users"  # 表名
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # 主键，自动递增
    session_id = Column(String(100), unique=True, index=True)   # 对话 ID，不重复
    name = Column(String(50), default="")                        # 姓名
    phone = Column(String(20), default="")                       # 手机号
    email = Column(String(100), default="")                      # 邮箱
    address = Column(String(200), default="")                    # 地址
    company = Column(String(100), default="")                    # 公司
    needs = Column(Text, default="")                             # 需求
    other = Column(Text, default="")                             # 其他信息
    created_at = Column(DateTime, default=datetime.now)          # 创建时间
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)  # 更新时间
    
    def to_dict(self) -> dict:
        """把对象转换成字典，方便返回 JSON"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "address": self.address,
            "company": self.company,
            "needs": self.needs,
            "other": self.other,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Message(Base):
    """
    消息记录表
    保存所有对话记录
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), index=True)    # 关联到用户的 session_id
    role = Column(String(20))                        # "user" 或 "assistant"
    content = Column(Text)                           # 消息内容
    timestamp = Column(DateTime, default=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ============================================================
# CRM 服务类
# ============================================================

class CRMService:
    """CRM 服务类：管理用户信息和对话记录"""
    
    def __init__(self):
        """初始化数据库连接"""
        from config import DATABASE_URL
        
        # 创建数据库引擎
        # echo=False：不打印 SQL 语句（True 时会打印，方便调试）
        self.engine = create_engine(DATABASE_URL, echo=False)
        
        # 创建所有表（如果不存在就创建）
        Base.metadata.create_all(self.engine)
        
        # 创建 Session 工厂
        # Session 是操作数据库的"会话"，相当于打开一个数据库连接
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info("✅ CRM 数据库初始化成功（SQLite）")
    
    def _get_db(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()
    
    # ---- 用户信息管理 ----
    
    def get_or_create_user(self, session_id: str) -> User:
        """
        获取或创建用户
        如果 session_id 已存在就返回已有用户，否则创建新用户
        """
        db = self._get_db()
        try:
            # 查询用户
            user = db.query(User).filter(User.session_id == session_id).first()
            
            if not user:
                # 用户不存在，创建新用户
                user = User(session_id=session_id)
                db.add(user)      # 添加到数据库
                db.commit()       # 提交（保存）
                db.refresh(user)  # 刷新（获取自动生成的 id 等）
                logger.info(f"✅ 创建新用户：{session_id}")
            
            return user
        finally:
            db.close()  # 关闭连接（重要！）
    
    def update_user_info(self, session_id: str, info: Dict[str, Any]) -> bool:
        """
        更新用户信息
        
        参数：
            session_id：用户的对话 ID
            info：要更新的信息字典
        """
        if not info:
            return False
        
        db = self._get_db()
        try:
            user = db.query(User).filter(User.session_id == session_id).first()
            
            if not user:
                user = User(session_id=session_id)
                db.add(user)
            
            # 更新字段（只更新有内容的字段）
            # 允许更新的字段列表（安全考虑，避免被改 id 之类的）
            allowed_fields = ["name", "phone", "email", "address", "company", "needs", "other"]
            
            updated = False
            for field, value in info.items():
                if field in allowed_fields and value:
                    # getattr：获取对象的属性
                    # setattr：设置对象的属性
                    old_value = getattr(user, field, "")
                    if not old_value:  # 只在原来为空时更新（不覆盖已有信息）
                        setattr(user, field, value)
                        updated = True
                        logger.info(f"📝 更新用户 {session_id} 的 {field}：{value}")
            
            if updated:
                user.updated_at = datetime.now()
                db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ 更新用户信息失败：{e}")
            db.rollback()  # 回滚（撤销失败的操作）
            return False
        finally:
            db.close()
    
    def get_all_users(self) -> List[Dict]:
        """获取所有用户列表"""
        db = self._get_db()
        try:
            users = db.query(User).order_by(User.updated_at.desc()).all()
            return [user.to_dict() for user in users]
        finally:
            db.close()
    
    def get_user(self, session_id: str) -> Optional[Dict]:
        """获取单个用户信息"""
        db = self._get_db()
        try:
            user = db.query(User).filter(User.session_id == session_id).first()
            return user.to_dict() if user else None
        finally:
            db.close()
    
    # ---- 消息记录管理 ----
    
    def save_message(self, session_id: str, role: str, content: str):
        """保存一条消息到数据库"""
        db = self._get_db()
        try:
            message = Message(
                session_id=session_id,
                role=role,
                content=content
            )
            db.add(message)
            db.commit()
        except Exception as e:
            logger.error(f"保存消息失败：{e}")
            db.rollback()
        finally:
            db.close()
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """获取某个会话的所有消息"""
        db = self._get_db()
        try:
            messages = (
                db.query(Message)
                .filter(Message.session_id == session_id)
                .order_by(Message.timestamp.asc())
                .all()
            )
            return [msg.to_dict() for msg in messages]
        finally:
            db.close()
    
    def get_session_text(self, session_id: str) -> str:
        """
        把某个会话的所有消息拼成一段文字
        用于给 LLM 提取信息
        """
        messages = self.get_session_messages(session_id)
        lines = []
        for msg in messages:
            role_name = "用户" if msg["role"] == "user" else "AI"
            lines.append(f"{role_name}：{msg['content']}")
        return "\n".join(lines)

    def export_readable_records(self, output_path: str) -> str:
        """
        导出可读的对话记录文本（UTF-8）
        按 session_id 分组，并按会话时间线排序，方便人工查看追溯。
        """
        db = self._get_db()
        try:
            rows = (
                db.query(Message)
                .order_by(Message.timestamp.asc(), Message.id.asc())
                .all()
            )

            grouped: Dict[str, List[Message]] = {}
            for row in rows:
                grouped.setdefault(row.session_id, []).append(row)

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            chunks: List[str] = []
            chunks.append("=== 小智对话记录导出 ===")
            chunks.append(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            chunks.append(f"会话总数: {len(grouped)}")
            chunks.append("")

            # 会话块按首条消息时间升序，保证导出文件是清晰的时间顺序
            session_items = sorted(
                grouped.items(),
                key=lambda item: (
                    item[1][0].timestamp if item[1] and item[1][0].timestamp else datetime.min,
                    item[1][0].id if item[1] else 0
                )
            )

            for sid, msgs in session_items:
                chunks.append("=" * 64)
                chunks.append(f"用户标识(session_id): {sid}")
                chunks.append(f"消息条数: {len(msgs)}")
                chunks.append("-" * 64)
                for m in msgs:
                    ts = m.timestamp.strftime("%Y-%m-%d %H:%M:%S") if m.timestamp else "-"
                    role_name = "用户" if m.role == "user" else "AI"
                    content = (m.content or "").replace("\r\n", "\n").replace("\r", "\n")
                    chunks.append(f"[{ts}] {role_name}")
                    chunks.append(content)
                    chunks.append("")
                chunks.append("")

            out.write_text("\n".join(chunks), encoding="utf-8")
            logger.info(f"✅ 已导出可读对话记录: {out}")
            return str(out)
        finally:
            db.close()


# 全局单例
_crm_instance = None

def get_crm_service() -> CRMService:
    """获取 CRM 服务实例"""
    global _crm_instance
    if _crm_instance is None:
        _crm_instance = CRMService()
    return _crm_instance
