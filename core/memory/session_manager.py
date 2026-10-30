import time
import logging
from typing import Dict, Optional
from langchain_classic.memory import ConversationBufferWindowMemory

logger = logging.getLogger("enhanced_medical_chatbot")

class SessionMemoryManager:
    """
    Quản lý bộ nhớ trò chuyện đa phiên (Multi-session Memory) độc lập cho từng người dùng/phiên.
    Hỗ trợ TTL tự động dọn dẹp các phiên không hoạt động để tránh tràn bộ nhớ RAM.
    """
    def __init__(self, ttl_seconds: int = 7200, k: int = 6):
        self.ttl_seconds = ttl_seconds
        self.k = k
        self.sessions: Dict[str, Dict] = {}

    def get_memory(self, session_id: Optional[str] = None) -> ConversationBufferWindowMemory:
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        current_time = time.time()
        self._cleanup_expired(current_time)

        if sid not in self.sessions:
            logger.info(f"Creating new conversation memory for session: {sid}")
            self.sessions[sid] = {
                "memory": ConversationBufferWindowMemory(
                    k=self.k,
                    memory_key="chat_history",
                    output_key="answer",
                    return_messages=True
                ),
                "last_active": current_time
            }
        else:
            self.sessions[sid]["last_active"] = current_time

        return self.sessions[sid]["memory"]

    def get_chat_history_str(self, session_id: Optional[str] = None) -> str:
        memory = self.get_memory(session_id)
        chat_history = ""
        if memory.chat_memory.messages:
            recent_messages = memory.chat_memory.messages[-(self.k * 2):]
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                    chat_history += f"{role}: {msg.content}\n"
        return chat_history

    def add_user_message(self, session_id: Optional[str], message: str):
        memory = self.get_memory(session_id)
        memory.chat_memory.add_user_message(message)

    def add_ai_message(self, session_id: Optional[str], message: str):
        memory = self.get_memory(session_id)
        memory.chat_memory.add_ai_message(message)

    def set_location(self, session_id: Optional[str], location: str, province: Optional[str] = None, district: Optional[str] = None):
        """Lưu địa điểm, tỉnh thành và quận huyện của người dùng vào phiên."""
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        self.get_memory(sid)
        if location and location.strip():
            self.sessions[sid]["location"] = location.strip()
            if province:
                self.sessions[sid]["province"] = province.strip()
            if district:
                self.sessions[sid]["district"] = district.strip()
            logger.info(f"Updated location for session '{sid}': {location.strip()} (prov={province}, dist={district})")

    def get_location(self, session_id: Optional[str]) -> Optional[str]:
        """Lấy chuỗi hiển thị địa điểm hiện tại của phiên."""
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        if sid in self.sessions:
            return self.sessions[sid].get("location")
        return None

    def get_location_details(self, session_id: Optional[str]):
        """Lấy chi tiết (location, province, district) của phiên."""
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        if sid in self.sessions:
            return (
                self.sessions[sid].get("location"),
                self.sessions[sid].get("province"),
                self.sessions[sid].get("district")
            )
        return (None, None, None)

    def set_last_doctor(self, session_id: Optional[str], doctor_info: Dict):
        """Lưu bác sĩ đang được trao đổi gần nhất trong phiên."""
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        self.get_memory(sid)
        self.sessions[sid]["last_doctor"] = doctor_info
        logger.info(f"Updated last_doctor for session '{sid}': {doctor_info.get('name', 'N/A')}")

    def get_last_doctor(self, session_id: Optional[str]) -> Optional[Dict]:
        """Lấy thông tin bác sĩ đang được trao đổi gần nhất trong phiên."""
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        if sid in self.sessions:
            return self.sessions[sid].get("last_doctor")
        return None

    def clear_session(self, session_id: Optional[str]):
        sid = session_id.strip() if session_id and session_id.strip() else "default_session"
        if sid in self.sessions:
            del self.sessions[sid]
            logger.info(f"Cleared session memory: {sid}")

    def _cleanup_expired(self, current_time: float):
        expired = [sid for sid, data in self.sessions.items() 
                   if current_time - data["last_active"] > self.ttl_seconds and sid != "default_session"]
        for sid in expired:
            del self.sessions[sid]
            logger.info(f"Cleaned up expired session memory: {sid}")
