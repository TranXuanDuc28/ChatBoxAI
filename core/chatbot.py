import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config.prompts import MEDICAL_SYSTEM_PROMPT
from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from core.classifier.medical_classifier import MedicalClassifier
from core.data.web_data_manager import WebDataManager
from core.llm.gemini import GeminiLLM, GeminiStreamingLLM
from core.memory.session_manager import SessionMemoryManager
from core.vector_store.embeddings import SentenceTransformerEmbeddings
from core.vector_store.enhanced_vector_store import EnhancedVectorStore
from utils.geo_helper import GeographicResolver, default_geo_resolver

logger = logging.getLogger("enhanced_medical_chatbot")

class EnhancedMedicalChatbot:
    def __init__(self):
        self.medical_classifier = MedicalClassifier()
        self.web_data_manager = WebDataManager()
        self.embeddings = SentenceTransformerEmbeddings()
        self.vector_store = EnhancedVectorStore(self.embeddings)
        self.llm = GeminiLLM()
        self.streaming_llm = GeminiStreamingLLM()
        self.session_manager = SessionMemoryManager(ttl_seconds=7200, k=6)
        self.setup_chains()
    
    def setup_chains(self):
        web_data = self.web_data_manager.get_web_data()
        self.vector_store.build_from_web_data(web_data)
        
        # 1. Tự động nạp động Tỉnh/Thành phố từ Database (Allcodes + Doctors + Clinics + Facilities)
        doctors = web_data.get("doctors", [])
        provinces_set = set()
        for prov_item in web_data.get("provinces", []):
            if isinstance(prov_item, dict):
                p_val = prov_item.get("valueVi") or prov_item.get("valueEn")
                if p_val and p_val != "N/A":
                    provinces_set.add(p_val.strip())
        for doc in doctors:
            prov = doc.get("province")
            if prov and prov != "N/A":
                provinces_set.add(prov.strip())
        for cl in web_data.get("clinics", []):
            cl_data = cl.get("clinicData", {})
            if isinstance(cl_data, dict):
                addr = cl_data.get("address", "")
                if addr:
                    _, p_match, _ = default_geo_resolver.resolve_location(addr)
                    if p_match:
                        provinces_set.add(p_match)
        for fac in web_data.get("facilities", []):
            addr = fac.get("address", "")
            if addr:
                _, p_match, _ = default_geo_resolver.resolve_location(addr)
                if p_match:
                    provinces_set.add(p_match)

        self.provinces = list(provinces_set)
        if not self.provinces:
            self.provinces = ["Hà Nội", "Hồ Chí Minh", "Đà Nẵng"]
        
        # Khởi tạo Dynamic Geo Resolver với các tỉnh nạp động từ DB + 63 tỉnh toàn quốc + mốc địa danh
        self.geo_resolver = GeographicResolver(db_provinces=self.provinces)
        logger.info(f"Automatically loaded {len(self.provinces)} dynamic provinces from database: {self.provinces}")
        
        # 2. Chuyên khoa
        specialties_set = set()
        for spec in web_data.get("specialties", []):
            name = spec.get("name")
            if name:
                specialties_set.add(name.strip())
        for doc in doctors:
            spec = doc.get("specialty")
            if spec and spec != "N/A":
                specialties_set.add(spec.strip())
        self.specialties = list(specialties_set)
        if not self.specialties:
            self.specialties = ["Cơ xương khớp", "Nhi khoa", "Da liễu", "Tim mạch", "Thần kinh", "Tiêu hóa", "Tai mũi họng", "Phụ khoa"]
        logger.info(f"Automatically loaded {len(self.specialties)} specialties from database: {self.specialties}")
    
    def refresh_data(self):
        logger.info("Refreshing web data...")
        self.web_data_manager.last_update = 0
        self.setup_chains()

    def clear_memory(self, session_id: Optional[str] = None):
        """Xóa lịch sử trò chuyện của một phiên cụ thể."""
        self.session_manager.clear_session(session_id)
    
    def get_health_status(self):
        """Get system health status"""
        return {
            "status": "healthy",
            "model": self.llm.model_name,
            "classifier_loaded": self.medical_classifier.model is not None,
            "vector_store_ready": self.vector_store.vectorstore is not None,
            "active_sessions": len(self.session_manager.sessions),
            "api_key_configured": bool(GEMINI_API_KEY)
        }

    def _is_obviously_medical(self, text: str) -> bool:
        """Kiểm tra xem câu hỏi có chứa từ ngữ y tế, triệu chứng, phòng khám, bác sĩ hoặc đặt lịch kể cả khi gõ không dấu."""
        from utils.geo_helper import remove_diacritics
        norm = remove_diacritics(text.lower())
        medical_keywords = [
            "kham", "bac si", "bs", "benh", "thuoc", "dau", "sot", "viem", "ho", "kho tho",
            "ung thu", "chuyen khoa", "phong kham", "benh vien", "dat lich", "gia kham",
            "chi phi", "tu van", "suc khoe", "xet nghiem", "sieu am", "chup", "chan thuong",
            "gay xuong", "moi met", "chong mat", "buon non", "nhiem trung", "dieu tri",
            "tai mui hong", "tim mach", "da lieu", "nha khoa", "san phu khoa", "noi tiet",
            "co xuong khop", "than kinh", "tieu hoa", "tiet nieu", "nam hoc", "nhi khoa"
        ]
        return any(kw in norm for kw in medical_keywords)

    def check_emergency(self, message: str) -> Optional[str]:
        """Kiểm tra dấu hiệu cấp cứu đỏ (Red Flags) để cảnh báo khẩn cấp ngay lập tức."""
        msg_lower = message.lower()
        emergency_keywords = [
            "bất tỉnh", "bat tinh", "ngất xỉu", "ngat xiu", "hôn mê", "hon me",
            "khó thở dữ dội", "kho tho du doi", "ngưng thở", "ngung tho",
            "nôn ra máu", "non ra mau", "ho ra máu", "ho ra mau", "xuất huyết",
            "đau thắt ngực dữ dội", "dau that nguc du doi", "đau ngực lan",
            "co giật", "co giat", "méo miệng", "meo mieng", "liệt nửa người", "liet nua nguoi",
            "tai biến", "đột quỵ", "dot quy"
        ]
        for kw in emergency_keywords:
            if kw in msg_lower:
                return (
                    f"🚨 **CẢNH BÁO Y TẾ KHẨN CẤP** 🚨\n\n"
                    f"Triệu chứng bạn vừa mô tả (**{kw}**) là dấu hiệu cảnh báo nguy hiểm cấp tính cần được can thiệp y tế ngay lập tức!\n\n"
                    f"👉 **HÀNH ĐỘNG NGAY**:\n"
                    f"1. **Gọi cấp cứu 115** hoặc nhờ người thân đưa bạn đến khoa **Cấp cứu Bệnh viện gần nhất** ngay lập tức.\n"
                    f"2. Tuyệt đối không tự ý uống thuốc hoặc trì hoãn chờ đợi.\n"
                    f"3. Giữ bình tĩnh, nằm nghỉ ngơi ở nơi thoáng khí trong khi chờ nhân viên y tế hỗ trợ."
                )
        return None

    def check_greeting(self, message: str) -> Optional[str]:
        """Kiểm tra và phản hồi câu chào hỏi thân thiện kèm giới thiệu năng lực hệ thống."""
        msg_clean = message.strip().lower()
        greetings = ["xin chào", "chào bạn", "chào bác sĩ", "chào bs", "hello", "hi", "alo", "hey", "bạn là ai", "hướng dẫn", "chức năng"]
        if msg_clean in greetings or (len(msg_clean) <= 15 and any(msg_clean.startswith(g) for g in ["chào", "xin chào", "hello", "hi"])):
            return (
                "👋 **Xin chào bạn! Tôi là Trợ lý Y tế Thông minh của Hệ thống Đặt lịch Khám bệnh.**\n\n"
                "Tôi có thể đồng hành và hỗ trợ bạn các vấn đề y tế sau:\n"
                "- 🩺 **Tư vấn & Phân tích triệu chứng:** Đánh giá tình trạng sức khỏe ban đầu, cảnh báo nguy cơ và gợi ý chuyên khoa phù hợp.\n"
                "- 👨‍⚕️ **Tra cứu Bác sĩ & Lịch khám:** Tìm bác sĩ giỏi theo chuyên khoa, khu vực (Hà Nội, TP.HCM, Đà Nẵng...) và xem lịch khám trực tuyến trong 7 ngày tới.\n"
                "- 🏥 **Gói khám sức khỏe:** Tra cứu các gói khám tổng quát, gói tầm soát bệnh lý, bảng giá và danh mục dịch vụ.\n"
                "- 🏢 **Bệnh viện & Phòng khám:** Cung cấp địa chỉ, tuyến viện, dịch vụ mũi nhọn và thời gian làm việc.\n\n"
                "Bạn đang có thắc mắc gì về sức khỏe hoặc cần hỗ trợ đặt lịch khám chuyên khoa nào, hãy chia sẻ cùng tôi nhé!"
            )
        return None
    def get_smart_context(self, user_message: str, target_province: Optional[str] = None, target_district: Optional[str] = None) -> str:
        """Lấy ngữ cảnh thông minh bằng Hybrid Search và Reranking với location boosting (Tỉnh thành & Quận huyện)."""
        if self.vector_store.vectorstore is None:
            return "Không có thông tin bổ sung"
        
        try:
            docs = self.vector_store.hybrid_search_with_rerank(
                user_message, 
                k=10, 
                target_province=target_province,
                target_district=target_district
            )
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.warning(f"Smart context error: {e}")
            try:
                return self._get_general_context(user_message)
            except Exception:
                return "Lỗi khi lấy thông tin ngữ cảnh"

    def _get_doctor_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh về bác sĩ."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke(user_message) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(user_message)
        
        # Ưu tiên docs về bác sĩ và chuyên khoa liên quan
        doctor_docs = [doc for doc in docs if doc.metadata.get("type") in ["doctor", "specialty", "specialty_detail"]]
        other_docs = [doc for doc in docs if doc.metadata.get("type") not in ["doctor", "specialty", "specialty_detail"]]
        
        # Kết hợp ưu tiên doctor_docs
        combined_docs = doctor_docs[:10] + other_docs[:5]
        
        return "\n".join([doc.page_content for doc in combined_docs])

    def _get_specialty_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh về chuyên khoa."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke(user_message) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(user_message)
        
        # Ưu tiên docs về chuyên khoa và bác sĩ thuộc chuyên khoa đó
        specialty_docs = [doc for doc in docs if doc.metadata.get("type") in ["specialty", "specialty_detail", "doctor"]]
        other_docs = [doc for doc in docs if doc.metadata.get("type") not in ["specialty", "specialty_detail", "doctor"]]
        
        combined_docs = specialty_docs[:12] + other_docs[:3]
        
        return "\n".join([doc.page_content for doc in combined_docs])

    def _get_symptom_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh khi người dùng mô tả triệu chứng."""
        # Tìm kiếm rộng hơn để bao gồm cả bác sĩ và chuyên khoa
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        docs = retriever.invoke(user_message) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(user_message)
        
        # Tìm thêm bác sĩ liên quan
        symptom_keywords = self._extract_symptom_keywords(user_message)
        if symptom_keywords:
            for keyword in symptom_keywords[:2]:  # Chỉ lấy 2 từ khóa chính
                extra_docs = retriever.invoke(f"bác sĩ chuyên khoa {keyword}") if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(f"bác sĩ chuyên khoa {keyword}")
                docs.extend(extra_docs[:3])
        
        # Loại bỏ duplicate và ưu tiên
        seen_content = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        return "\n".join([doc.page_content for doc in unique_docs[:15]])

    def _extract_symptom_keywords(self, message: str) -> List[str]:
        """
        Trích xuất chuyên khoa và triệu chứng theo Semantic Vector Search động 
        thay vì dựa vào từ điển hardcode tĩnh.
        """
        keywords = []
        msg_lower = message.lower()

        # 1. Kiểm tra trực tiếp các chuyên khoa hiện có trong Database
        if hasattr(self, 'specialties') and self.specialties:
            for spec in self.specialties:
                if spec.lower() in msg_lower:
                    keywords.append(spec)

        # 2. Sử dụng Dense Vector Search để tìm chuyên khoa tương đồng ngữ nghĩa nhất
        if hasattr(self, 'vector_store') and self.vector_store and self.vector_store.vectorstore:
            try:
                sim_docs = self.vector_store.vectorstore.similarity_search(message, k=4)
                for doc in sim_docs:
                    s_name = doc.metadata.get("specialty_name")
                    if s_name and s_name not in keywords:
                        keywords.append(s_name)
            except Exception as e:
                logger.warning(f"Semantic specialty extraction error: {e}")

        return keywords[:3]

    def _get_general_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh thông thường."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.invoke(user_message) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(user_message)
        return "\n".join([doc.page_content for doc in docs])
    
    def _fallback_simple_response(self, user_message: str) -> str:
        """
        Dùng bộ lọc từ khóa đơn giản để trả lời khi LLM bị lỗi/hết quota.
        """
        msg_lower = user_message.lower()
        response = "⚠️ **Hệ thống AI đang quá tải**: Tôi đang trả lời bạn bằng chế độ truy xuất dữ liệu trực tiếp (Dự phòng).\n\n"
        
        # 1. Tìm theo tên bác sĩ hoặc chuyên khoa bằng từ khóa
        doctors = []
        # Trích xuất tên bác sĩ đơn giản
        search_term = msg_lower.replace("bác sĩ", "").replace("bs", "").replace("lịch khám", "").replace("hôm nay", "").strip()
        
        # Ưu tiên tìm theo tên nếu search_term đủ dài
        if len(search_term) > 2:
            doctors = self.find_doctors_by_name(search_term)
        
        # Nếu không thấy theo tên, thử tìm theo chuyên khoa
        if not doctors:
            specs = getattr(self, 'specialties', [])
            for spec in specs:
                if spec.lower() in msg_lower:
                    doctors = self.find_doctors_by_specialty(spec, limit=3)
                    break

        if doctors:
            response += "Tôi tìm thấy thông tin sau trong hệ thống:\n\n"
            for doc in doctors:
                response += f"👨‍⚕️ **Bác sĩ {doc['name']}**\n"
                response += f"- Chuyên khoa: {doc['specialty']}\n"
                response += f"- Phòng khám: {doc['clinic']}\n"
                # Lấy lịch khám nếu có
                full_data = doc.get("full_data", {})
                schedules = full_data.get("schedules", [])
                if schedules:
                    response += f"- Lịch khám: {'; '.join(schedules)}\n"
                response += f"- Giá khám: {doc.get('price', 'N/A')}\n"
                response += "\n"
            return response

        # 2. Cuối cùng, nếu ko khớp gì thì lấy ngữ cảnh thô từ VectorDB
        context = self.get_smart_context(user_message)
        if context and len(context) > 20:
            response += "Dữ liệu liên quan từ hệ thống:\n"
            # Làm sạch context một chút
            clean_context = context.replace("\n\n", "\n").strip()
            response += clean_context[:1000]
            return response
            
        return "Xin lỗi, AI đang bận và tôi không tìm thấy dữ liệu phù hợp trong chế độ dự phòng. Bạn vui lòng thử lại sau vài phút."

    def _clean_response_links(self, response_text: str) -> str:
        """Đảm bảo mọi link đặt lịch đều là đường dẫn tương đối chuẩn /detail-doctor/{id} và loại bỏ tàn dư thừa."""
        if not response_text:
            return response_text
        cleaned = re.sub(r'https?://localhost:\d+/detail-doctor/', '/detail-doctor/', response_text)
        cleaned = re.sub(r'https?://127\.0\.0\.1:\d+/detail-doctor/', '/detail-doctor/', cleaned)
        # Loại bỏ các tàn dư lửng lơ như "Khi bấm vào..." ở cuối câu
        cleaned = re.sub(r'\n+\s*Khi bấm vào(?:\s+link[^\n]*)?$', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _enrich_context_with_doctor_schedule(self, user_message: str, chat_history: str, session_id: str, base_context: str) -> str:
        """
        Phát hiện nếu người dùng đang hỏi về lịch khám, ngày giờ, ca khám hoặc follow-up về một bác sĩ,
        tự động tìm thông tin chi tiết và bảng lịch 7 ngày của bác sĩ đó để đưa trực tiếp lên đầu ngữ cảnh.
        """
        web_data = self.web_data_manager.get_web_data()
        all_doctors = web_data.get("doctors", [])
        if not all_doctors:
            return base_context

        msg_lower = user_message.lower()
        history_lower = chat_history.lower() if chat_history else ""

        # 1. Tìm xem bác sĩ nào được nhắc đến trực tiếp trong câu hỏi
        matched_doctor = None
        for doc in all_doctors:
            d_name = doc.get("name", "").lower()
            d_last = doc.get("lastName", "").lower()
            if (d_name and d_name in msg_lower) or (d_last and len(d_last) > 2 and d_last in msg_lower):
                matched_doctor = doc
                break

        # 2. Nếu không có tên trực tiếp trong câu hỏi, kiểm tra xem có phải câu hỏi follow-up (lịch/giờ/mai/hôm nay/giá/bác sĩ)
        schedule_keywords = ["lịch", "khám", "mai", "hôm nay", "giờ", "mấy giờ", "thứ", "trống", "cụ thể", "chi phí", "giá", "bác sĩ"]
        is_followup = any(kw in msg_lower for kw in schedule_keywords)

        if not matched_doctor and is_followup:
            # Kiểm tra doctor đã lưu trong session
            last_doc = self.session_manager.get_last_doctor(session_id)
            if last_doc:
                for doc in all_doctors:
                    if doc.get("id") == last_doc.get("id"):
                        matched_doctor = doc
                        break

            # Nếu chưa có trong session, quét ngược trong lịch sử hội thoại gần nhất
            if not matched_doctor and history_lower:
                for doc in all_doctors:
                    d_name = doc.get("name", "").lower()
                    d_last = doc.get("lastName", "").lower()
                    if (d_name and d_name in history_lower) or (d_last and len(d_last) > 2 and d_last in history_lower):
                        matched_doctor = doc
                        break

        # 3. Nếu tìm thấy bác sĩ liên quan, lưu vào session và inject khối thông tin lịch khám ưu tiên
        if matched_doctor:
            self.session_manager.set_last_doctor(session_id, matched_doctor)
            schedules_list = matched_doctor.get("schedules", [])
            schedules_str = "\n".join(schedules_list) if schedules_list else "Hiện tại chưa cập nhật lịch khám trực tuyến trong 7 ngày tới."
            doc_id = matched_doctor.get('id')
            booking_link = f"/detail-doctor/{doc_id}" if doc_id else ""

            doctor_schedule_block = f"""
================================================================================
THÔNG TIN LỊCH KHÁM THỜI GIAN THỰC CỦA BÁC SĨ (ƯU TIÊN TUYỆT ĐỐI):
👨‍⚕️ Bác sĩ: {matched_doctor.get('name')} (ID: {doc_id})
🏥 Chuyên khoa: {matched_doctor.get('specialty')}
🏢 Phòng khám / Cơ sở: {matched_doctor.get('clinic')} (Địa chỉ: {matched_doctor.get('addressClinic', matched_doctor.get('province', ''))})
💰 Giá khám: {matched_doctor.get('price')}
🔗 Link đặt lịch chính thức: 👉 [Bấm vào đây để chọn ngày & giờ khám với Bác sĩ {matched_doctor.get('name')}]( {booking_link} )

📅 BẢNG LỊCH KHÁM 7 NGÀY TỚI THEO THỜI GIAN THỰC (ĐỐI CHIẾU CHÍNH XÁC):
{schedules_str}
================================================================================
"""
            return doctor_schedule_block + "\n\n" + base_context

        return base_context

    # Cập nhật phương thức generate_response để sử dụng smart context, multi-session và fallback
    def generate_response(self, user_message: str, session_id: Optional[str] = "default_session", location: Optional[str] = None) -> str:
        """
        Tạo phản hồi không streaming với ngữ cảnh thông minh, quản lý phiên và nhận diện vị trí.
        """
        try:
            # 1. Kiểm tra trường hợp cấp cứu khẩn cấp (Red Flags)
            emergency_alert = self.check_emergency(user_message)
            if emergency_alert:
                self.session_manager.add_user_message(session_id, user_message)
                self.session_manager.add_ai_message(session_id, emergency_alert)
                return emergency_alert

            # 2. Kiểm tra câu chào hỏi / giới thiệu ban đầu
            greeting_resp = self.check_greeting(user_message)
            if greeting_resp:
                self.session_manager.add_user_message(session_id, user_message)
                self.session_manager.add_ai_message(session_id, greeting_resp)
                return greeting_resp

            # Lấy lịch sử trò chuyện theo đúng phiên
            chat_history = self.session_manager.get_chat_history_str(session_id)
            has_history = bool(chat_history and chat_history.strip())

            # 3. Sử dụng PhoBERT để kiểm duyệt nếu câu hỏi hoàn toàn phi y tế (chỉ kiểm duyệt câu mở đầu)
            if not has_history and hasattr(self, 'medical_classifier') and self.medical_classifier:
                try:
                    is_med, conf = self.medical_classifier.is_medical_question(user_message)
                    if not is_med and conf > 0.85:
                        if not self._is_obviously_medical(user_message):
                            out_of_scope = (
                                "Xin lỗi bạn, tôi là Trợ lý Y tế Thông minh và chỉ được đào tạo chuyên sâu về "
                                "lĩnh vực y tế, sức khỏe, bác sĩ và đặt lịch khám bệnh. Tôi không thể hỗ trợ các chủ đề ngoài y tế.\n\n"
                                "Nếu bạn có bất kỳ câu hỏi nào về sức khỏe, triệu chứng bệnh hoặc cần tìm bác sĩ/gói khám, "
                                "xin vui lòng hỏi tôi nhé!"
                            )
                            self.session_manager.add_user_message(session_id, user_message)
                            self.session_manager.add_ai_message(session_id, out_of_scope)
                            return out_of_scope
                        else:
                            logger.info(f"PhoBERT flagged non-medical, but medical indicator matched. Proceeding with query.")
                except Exception as cf_err:
                    logger.warning(f"Classification check skipped: {cf_err}")

            # Xử lý nhận diện và ghi nhớ vị trí (Location Intelligence: District & Province)
            active_loc_display = None
            active_province = None
            active_district = None

            if location and isinstance(location, str) and location.strip():
                f_disp, f_prov, f_dist = self._detect_location_details(location)
                active_loc_display = f_disp or location.strip()
                active_province = f_prov
                active_district = f_dist
                self.session_manager.set_location(session_id, active_loc_display, active_province, active_district)

            msg_disp, msg_prov, msg_dist = self._detect_location_details(user_message)
            if msg_disp or msg_prov or msg_dist:
                active_loc_display = msg_disp or active_loc_display
                active_province = msg_prov or active_province
                active_district = msg_dist or active_district
                self.session_manager.set_location(session_id, active_loc_display, active_province, active_district)

            if not active_loc_display:
                active_loc_display, active_province, active_district = self.session_manager.get_location_details(session_id)

            location_desc = active_loc_display if active_loc_display else "Chưa rõ vị trí (Cần hỏi người bệnh quận/huyện, tỉnh/thành phố để gợi ý bác sĩ gần nhất)"

            # 4. Lấy ngữ cảnh thông minh có ưu tiên vị trí và quận/huyện
            context = self.get_smart_context(user_message, target_province=active_province, target_district=active_district)
            context = self._enrich_context_with_doctor_schedule(user_message, chat_history, session_id, context)

            now = datetime.now()
            day_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
            today_weekday = day_names[now.weekday()]
            tomorrow = now + timedelta(days=1)
            tomorrow_weekday = day_names[tomorrow.weekday()]

            current_date_str = f"{today_weekday}, ngày {now.strftime('%d/%m/%Y')}"
            tomorrow_date_str = f"{tomorrow_weekday}, ngày {tomorrow.strftime('%d/%m/%Y')}"

            # 6. Tạo prompt với MEDICAL_SYSTEM_PROMPT
            prompt = MEDICAL_SYSTEM_PROMPT.format(
                current_date_str=current_date_str,
                tomorrow_date_str=tomorrow_date_str,
                context=context,
                chat_history=chat_history,
                patient_location=location_desc,
                question=user_message
            )

            # 7. Gọi LLM để tạo phản hồi
            response = self.llm.invoke(prompt)
            if isinstance(response, dict) and "text" in response:
                response = response["text"]
            elif hasattr(response, 'content'):
                response = response.content
            else:
                response = str(response)

            response = self._clean_response_links(response)

            # Cập nhật bộ nhớ trò chuyện của phiên
            try:
                self.session_manager.add_user_message(session_id, user_message)
                self.session_manager.add_ai_message(session_id, response)
            except Exception as e:
                logger.warning(f"Memory update error: {e}")

            return response
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            fallback_res = self._fallback_simple_response(user_message)
            self.session_manager.add_user_message(session_id, user_message)
            self.session_manager.add_ai_message(session_id, fallback_res)
            return fallback_res

    def generate_streaming_response(self, user_message: str, session_id: Optional[str] = "default_session", location: Optional[str] = None):
        """
        Tạo phản hồi streaming với ngữ cảnh thông minh, quản lý phiên và nhận diện vị trí.
        """
        try:
            # 1. Kiểm tra trường hợp cấp cứu khẩn cấp (Red Flags)
            emergency_alert = self.check_emergency(user_message)
            if emergency_alert:
                yield emergency_alert
                self.session_manager.add_user_message(session_id, user_message)
                self.session_manager.add_ai_message(session_id, emergency_alert)
                return

            # 2. Kiểm tra câu chào hỏi / giới thiệu ban đầu
            greeting_resp = self.check_greeting(user_message)
            if greeting_resp:
                yield greeting_resp
                self.session_manager.add_user_message(session_id, user_message)
                self.session_manager.add_ai_message(session_id, greeting_resp)
                return

            # Lấy lịch sử trò chuyện theo đúng phiên
            chat_history = self.session_manager.get_chat_history_str(session_id)
            has_history = bool(chat_history and chat_history.strip())

            # 3. Sử dụng PhoBERT để kiểm duyệt nếu câu hỏi hoàn toàn phi y tế (chỉ kiểm duyệt câu mở đầu)
            if not has_history and hasattr(self, 'medical_classifier') and self.medical_classifier:
                try:
                    is_med, conf = self.medical_classifier.is_medical_question(user_message)
                    if not is_med and conf > 0.85:
                        if not self._is_obviously_medical(user_message):
                            out_of_scope = (
                                "Xin lỗi bạn, tôi là Trợ lý Y tế Thông minh và chỉ được đào tạo chuyên sâu về "
                                "lĩnh vực y tế, sức khỏe, bác sĩ và đặt lịch khám bệnh. Tôi không thể hỗ trợ các chủ đề ngoài y tế.\n\n"
                                "Nếu bạn có bất kỳ câu hỏi nào về sức khỏe, triệu chứng bệnh hoặc cần tìm bác sĩ/gói khám, "
                                "xin vui lòng hỏi tôi nhé!"
                            )
                            yield out_of_scope
                            self.session_manager.add_user_message(session_id, user_message)
                            self.session_manager.add_ai_message(session_id, out_of_scope)
                            return
                        else:
                            logger.info(f"PhoBERT flagged non-medical, but medical indicator matched. Proceeding with streaming.")
                except Exception as cf_err:
                    logger.warning(f"Classification check skipped: {cf_err}")

            # Xử lý nhận diện và ghi nhớ vị trí (Location Intelligence: District & Province)
            active_loc_display = None
            active_province = None
            active_district = None

            if location and isinstance(location, str) and location.strip():
                f_disp, f_prov, f_dist = self._detect_location_details(location)
                active_loc_display = f_disp or location.strip()
                active_province = f_prov
                active_district = f_dist
                self.session_manager.set_location(session_id, active_loc_display, active_province, active_district)

            msg_disp, msg_prov, msg_dist = self._detect_location_details(user_message)
            if msg_disp or msg_prov or msg_dist:
                active_loc_display = msg_disp or active_loc_display
                active_province = msg_prov or active_province
                active_district = msg_dist or active_district
                self.session_manager.set_location(session_id, active_loc_display, active_province, active_district)

            if not active_loc_display:
                active_loc_display, active_province, active_district = self.session_manager.get_location_details(session_id)

            location_desc = active_loc_display if active_loc_display else "Chưa rõ vị trí (Cần hỏi người bệnh quận/huyện, tỉnh/thành phố để gợi ý bác sĩ gần nhất)"

            # 4. Sử dụng smart context có ưu tiên vị trí và quận/huyện
            context = self.get_smart_context(user_message, target_province=active_province, target_district=active_district)
            context = self._enrich_context_with_doctor_schedule(user_message, chat_history, session_id, context)

            now = datetime.now()
            day_names = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
            today_weekday = day_names[now.weekday()]
            tomorrow = now + timedelta(days=1)
            tomorrow_weekday = day_names[tomorrow.weekday()]

            current_date_str = f"{today_weekday}, ngày {now.strftime('%d/%m/%Y')}"
            tomorrow_date_str = f"{tomorrow_weekday}, ngày {tomorrow.strftime('%d/%m/%Y')}"

            # 6. Tạo prompt
            prompt = MEDICAL_SYSTEM_PROMPT.format(
                current_date_str=current_date_str,
                tomorrow_date_str=tomorrow_date_str,
                context=context,
                chat_history=chat_history,
                patient_location=location_desc,
                question=user_message
            )
            
            # 7. Stream response từ LLM
            full_response = ""
            for chunk in self.streaming_llm.stream_response(prompt):
                chunk_clean = chunk.replace("http://localhost:3000/detail-doctor/", "/detail-doctor/").replace("http://127.0.0.1:3000/detail-doctor/", "/detail-doctor/")
                full_response += chunk_clean
                yield chunk_clean
            
            full_response = self._clean_response_links(full_response)

            # Cập nhật memory của phiên với response đầy đủ
            try:
                self.session_manager.add_user_message(session_id, user_message)
                self.session_manager.add_ai_message(session_id, full_response)
            except Exception as e:
                logger.warning(f"Memory update error: {e}")
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            fallback_text = self._fallback_simple_response(user_message)
            yield fallback_text
            self.session_manager.add_user_message(session_id, user_message)
            self.session_manager.add_ai_message(session_id, fallback_text)

    def _detect_location_details(self, message: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Phát hiện chi tiết vị trí (chuỗi hiển thị, tỉnh/thành, quận/huyện)
        thông qua Dynamic Geographic Resolver (Hỗ trợ 63 tỉnh thành, các quận huyện, mốc địa danh nổi tiếng).
        """
        if not hasattr(self, 'geo_resolver') or not self.geo_resolver:
            self.geo_resolver = default_geo_resolver
        return self.geo_resolver.resolve_location(message)

    def _detect_province_from_message(self, message: str) -> Optional[str]:
        """Tự động phát hiện tỉnh thành từ câu hỏi hoặc địa chỉ của người dùng (tương thích ngược)."""
        full_display, province, _ = self._detect_location_details(message)
        return province or full_display

    def _get_doctor_suggestions_for_streaming(self, user_message: str) -> List[str]:
        """
        Tạo gợi ý bác sĩ dưới dạng chunks cho streaming.
        """
        try:
            message_lower = user_message.lower()
            suggestions = []
            
            # Nếu hỏi về triệu chứng (bỏ từ 'khám' để tránh nhầm với hỏi lịch khám thông thường)
            if any(word in message_lower for word in ["đau", "bệnh", "triệu chứng", "mỏi", "nhức", "sốt", "ho", "mệt", "ngứa"]):
                doctors = self.suggest_doctors_for_symptoms(user_message, limit=3)
                if doctors:
                    suggestions.append("🩺 **Gợi ý bác sĩ phù hợp:**\n")
                    for i, doctor in enumerate(doctors, 1):
                        doctor_info = f"{i}. **Bác sĩ {doctor['name']}** - {doctor['specialty']}\n"
                        doctor_info += f"   📍 {doctor['clinic']}, {doctor['province']}\n"
                        doctor_info += f"   💰 {doctor['price']}\n"
                        if doctor.get('suggested_reason'):
                            doctor_info += f"   ✨ {doctor['suggested_reason']}\n"
                        doctor_info += "\n"
                        suggestions.append(doctor_info)
            
            # Nếu hỏi về chuyên khoa
            elif any(word in message_lower for word in ["chuyên khoa", "khoa"]):
                detected_specialty = None
                specs = getattr(self, 'specialties', [])
                for spec in specs:
                    if spec.lower() in message_lower:
                        detected_specialty = spec
                        break
                
                if detected_specialty:
                    doctors = self.find_doctors_by_specialty(detected_specialty, limit=5)
                    
                    # Lọc theo tỉnh thành nếu có trong câu hỏi
                    target_province = self._detect_province_from_message(user_message)
                    if target_province:
                        doctors = [d for d in doctors if target_province.lower() in d.get("province", "").lower()]
                        
                    doctors = doctors[:3]
                    if doctors:
                        suggestions.append(f"👨‍⚕️ **Các bác sĩ chuyên khoa {detected_specialty}:**\n")
                        for j, doctor in enumerate(doctors, 1):
                            doctor_info = f"{j}. **Bác sĩ {doctor['name']}**\n"
                            doctor_info += f"   📍 {doctor['clinic']}, {doctor['province']}\n"
                            doctor_info += f"   💰 {doctor['price']}\n\n"
                            suggestions.append(doctor_info)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"Error generating doctor suggestions for streaming: {e}")
            return []

    # Thêm method để debug và monitoring
    def get_system_stats(self) -> Dict:
        """
        Lấy thống kê hệ thống để monitoring.
        """
        stats = self.get_health_status()
        
        # Thêm thông tin về vector store
        if self.vector_store and self.vector_store.vectorstore:
            try:
                # Đếm số documents theo loại
                all_docs = self.vector_store.vectorstore.docstore._dict
                type_counts = {}
                
                for doc_id, doc in all_docs.items():
                    doc_type = doc.metadata.get("type", "unknown")
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                
                stats["vector_store_stats"] = {
                    "total_documents": len(all_docs),
                    "document_types": type_counts,
                    "specialty_mapping_size": len(getattr(self.vector_store, 'specialty_mapping', {}))
                }
            except Exception as e:
                stats["vector_store_stats"] = {"error": str(e)}
        
        return stats

    # Thêm method hỗ trợ để debug context
    def debug_context_selection(self, user_message: str) -> Dict:
        """
        Debug method để xem AI đang chọn context như thế nào.
        """
        return {
            "message": user_message,
            "detected_intent": "semantic_search",
            "context_strategy": "vector_database_rag"
        }
    # Thêm vào class EnhancedMedicalChatbot

    def find_doctors_by_name(self, doctor_name: str) -> List[Dict]:
        """
        Tìm bác sĩ theo tên cụ thể.
        """
        if not self.vector_store.vectorstore:
            return []
        
        try:
            # Tìm kiếm với query cụ thể
            docs = self.vector_store.vectorstore.similarity_search(
                f"bác sĩ {doctor_name}",
                k=10
            )
            
            # Filter chỉ lấy documents về bác sĩ
            doctor_docs = []
            for doc in docs:
                if doc.metadata.get("type") == "doctor":
                    doctor_data = doc.metadata.get("data", {})
                    doctor_name_in_doc = f"{doctor_data.get('firstName', '')} {doctor_data.get('lastName', '')}".strip()
                    
                    # Kiểm tra độ tương đồng tên
                    if doctor_name.lower() in doctor_name_in_doc.lower() or doctor_name_in_doc.lower() in doctor_name.lower():
                        doctor_docs.append({
                            "name": doctor_name_in_doc,
                            "specialty": doctor_data.get("specialty", "N/A"),
                            "clinic": doctor_data.get("clinic", "N/A"),
                            "province": doctor_data.get("province", "N/A"),
                            "price": doctor_data.get("price", "N/A"),
                            "payment": doctor_data.get("payment", "N/A"),
                            "note": doctor_data.get("note", "")[:200],
                            "full_data": doctor_data
                        })
            
            return doctor_docs
        
        except Exception as e:
            logger.error(f"Error finding doctors by name: {e}")
            return []

    def find_doctors_by_specialty(self, specialty_name: str, limit: int = 5) -> List[Dict]:
        """
        Tìm bác sĩ theo chuyên khoa cụ thể.
        """
        if not self.vector_store.vectorstore:
            return []
        
        try:
            # Tìm kiếm với nhiều query variant
            search_queries = [
                f"chuyên khoa {specialty_name}",
                f"bác sĩ {specialty_name}",
                f"{specialty_name} bác sĩ",
                specialty_name
            ]
            
            all_docs = []
            for query in search_queries:
                docs = self.vector_store.vectorstore.similarity_search(query, k=8)
                all_docs.extend(docs)
            
            # Lọc và xử lý documents
            doctor_matches = []
            seen_doctors = set()
            
            for doc in all_docs:
                if doc.metadata.get("type") == "doctor":
                    doctor_data = doc.metadata.get("data", {})
                    doctor_id = doctor_data.get("id")
                    doctor_specialty = doctor_data.get("specialty", "").lower()
                    
                    # Tránh duplicate
                    if doctor_id in seen_doctors:
                        continue
                    seen_doctors.add(doctor_id)
                    
                    # Kiểm tra độ tương đồng chuyên khoa
                    if (specialty_name.lower() in doctor_specialty or 
                        doctor_specialty in specialty_name.lower() or
                        any(keyword in doctor_specialty for keyword in specialty_name.lower().split())):
                        
                        doctor_name = f"{doctor_data.get('firstName', '')} {doctor_data.get('lastName', '')}".strip()
                        doctor_matches.append({
                            "name": doctor_name,
                            "specialty": doctor_data.get("specialty", "N/A"),
                            "clinic": doctor_data.get("clinic", "N/A"),
                            "province": doctor_data.get("province", "N/A"),
                            "price": doctor_data.get("price", "N/A"),
                            "payment": doctor_data.get("payment", "N/A"),
                            "note": doctor_data.get("note", "")[:200],
                            "relevance_score": self._calculate_specialty_relevance(specialty_name, doctor_specialty),
                            "full_data": doctor_data
                        })
            
            # Sắp xếp theo độ liên quan
            doctor_matches.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return doctor_matches[:limit]
        
        except Exception as e:
            logger.error(f"Error finding doctors by specialty: {e}")
            return []

    def _calculate_specialty_relevance(self, query_specialty: str, doctor_specialty: str) -> float:
        """
        Tính điểm độ liên quan giữa chuyên khoa query và chuyên khoa của bác sĩ.
        """
        query_lower = query_specialty.lower()
        doctor_lower = doctor_specialty.lower()
        
        # Exact match
        if query_lower == doctor_lower:
            return 1.0
        
        # Substring match
        if query_lower in doctor_lower or doctor_lower in query_lower:
            return 0.8
        
        # Word overlap
        query_words = set(query_lower.split())
        doctor_words = set(doctor_lower.split())
        
        if query_words & doctor_words:  # Có từ chung
            overlap_ratio = len(query_words & doctor_words) / len(query_words | doctor_words)
            return 0.6 * overlap_ratio
        
        return 0.0

    def get_specialty_overview(self, specialty_name: str) -> Dict:
        """
        Lấy thông tin tổng quan về một chuyên khoa, bao gồm mô tả và danh sách bác sĩ.
        """
        if not self.vector_store.vectorstore:
            return {"error": "Vector store not available"}
        
        try:
            # Tìm thông tin về chuyên khoa
            specialty_docs = self.vector_store.vectorstore.similarity_search(
                f"chuyên khoa {specialty_name}",
                k=10
            )
            
            # Tìm bác sĩ thuộc chuyên khoa
            doctors = self.find_doctors_by_specialty(specialty_name, limit=10)
            
            # Tìm thông tin chi tiết về chuyên khoa
            specialty_info = ""
            for doc in specialty_docs:
                if doc.metadata.get("type") in ["specialty", "specialty_detail"]:
                    specialty_info += doc.page_content + "\n"
            
            return {
                "specialty_name": specialty_name,
                "description": specialty_info.strip() if specialty_info else "Không có thông tin chi tiết",
                "doctors_count": len(doctors),
                "doctors": doctors,
                "related_info": [doc.page_content for doc in specialty_docs[:5] 
                            if doc.metadata.get("type") not in ["doctor"]]
            }
        
        except Exception as e:
            logger.error(f"Error getting specialty overview: {e}")
            return {"error": str(e)}

    def _extract_symptom_keywords(self, symptoms: str) -> List[str]:
        """
        Sử dụng LLM để trích xuất các chuyên khoa phù hợp từ triệu chứng.
        """
        prompt = f"""Phân tích các triệu chứng sau và trả về danh sách tên các chuyên khoa y tế phù hợp nhất (ví dụ: Cơ xương khớp, Nhi khoa, Da liễu...).
        
        Triệu chứng: {symptoms}
        
        Trả về kết quả dưới dạng danh sách tên chuyên khoa, cách nhau bởi dấu phẩy. Không giải thích gì thêm.
        Ví dụ: Cơ xương khớp, Thần kinh"""
        
        try:
            response = self.llm.invoke(prompt)
            if isinstance(response, dict) and "text" in response:
                content = response["text"]
            else:
                content = str(response)
                
            specialties = [s.strip() for s in content.split(",") if s.strip()]
            return specialties
        except Exception as e:
            logger.error(f"Error extracting symptom keywords: {e}")
            # Fallback cơ bản nếu API bị lỗi/quota
            message_lower = symptoms.lower()
            if any(word in message_lower for word in ["xương", "khớp", "đau chân", "đau tay", "lưng"]):
                return ["Cơ xương khớp"]
            return []

    def suggest_doctors_for_symptoms(self, symptoms: str, limit: int = 3) -> List[Dict]:
        """
        Đề xuất bác sĩ dựa trên triệu chứng mô tả.
        """
        try:
            # Trích xuất keywords từ triệu chứng
            symptom_keywords = self._extract_symptom_keywords(symptoms)
            
            if not symptom_keywords:
                return []
            
            # Kiểm tra xem có yêu cầu về tỉnh thành không
            target_province = self._detect_province_from_message(symptoms)
            
            suggested_doctors = []
            
            # Tìm bác sĩ cho từng chuyên khoa liên quan
            for specialty in symptom_keywords:
                # Lấy số lượng nhiều hơn để lọc theo tỉnh thành
                doctors = self.find_doctors_by_specialty(specialty, limit=10)
                
                # Nếu có tỉnh thành đích, lọc theo tỉnh thành
                if target_province:
                    doctors = [d for d in doctors if target_province.lower() in d.get("province", "").lower()]
                
                for doctor in doctors:
                    doctor["suggested_reason"] = f"Chuyên khoa {specialty} phù hợp với triệu chứng của bạn"
                    suggested_doctors.append(doctor)
            
            # Loại bỏ duplicate và giới hạn số lượng
            seen_doctors = set()
            unique_doctors = []
            
            for doctor in suggested_doctors:
                doctor_id = doctor.get("full_data", {}).get("id")
                if doctor_id not in seen_doctors:
                    seen_doctors.add(doctor_id)
                    unique_doctors.append(doctor)
            
            return unique_doctors[:limit]
        
        except Exception as e:
            logger.error(f"Error suggesting doctors for symptoms: {e}")
            return []

    # Method để tích hợp vào generate_response
    def enhance_response_with_doctor_suggestions(self, user_message: str, base_response: str) -> str:
        """
        Tăng cường phản hồi với gợi ý bác sĩ cụ thể nếu phù hợp.
        """
        message_lower = user_message.lower()
        
        # Nếu người dùng hỏi về triệu chứng (bỏ từ 'khám' để tránh nhầm với hỏi lịch khám thông thường)
        if any(word in message_lower for word in ["đau", "bệnh", "triệu chứng", "mỏi", "nhức", "sốt", "ho", "mệt", "ngứa"]):
            doctors = self.suggest_doctors_for_symptoms(user_message, limit=3)
            if doctors:
                suggestion_text = "\n\n🩺 **Gợi ý bác sĩ phù hợp:**\n"
                for i, doctor in enumerate(doctors, 1):
                    suggestion_text += f"{i}. **Bác sĩ {doctor['name']}** - {doctor['specialty']}\n"
                    suggestion_text += f"   📍 {doctor['clinic']}, {doctor['province']}\n"
                    suggestion_text += f"   💰 {doctor['price']}\n"
                    if doctor.get('suggested_reason'):
                        suggestion_text += f"   ✨ {doctor['suggested_reason']}\n"
                    suggestion_text += "\n"
                
                return base_response + suggestion_text
        
        # Nếu người dùng hỏi về chuyên khoa cụ thể
        elif any(word in message_lower for word in ["chuyên khoa", "khoa"]):
            detected_specialty = None
            specs = getattr(self, 'specialties', [])
            for spec in specs:
                if spec.lower() in message_lower:
                    detected_specialty = spec
                    break
            
            if detected_specialty:
                doctors = self.find_doctors_by_specialty(detected_specialty, limit=5)
                
                # Lọc theo tỉnh thành nếu có trong câu hỏi
                target_province = self._detect_province_from_message(user_message)
                if target_province:
                    doctors = [d for d in doctors if target_province.lower() in d.get("province", "").lower()]
                
                doctors = doctors[:3]
                if doctors:
                    suggestion_text = f"\n\n👨‍⚕️ **Các bác sĩ chuyên khoa {detected_specialty}:**\n"
                    for idx, doctor in enumerate(doctors, 1):
                        suggestion_text += f"{idx}. **Bác sĩ {doctor['name']}**\n"
                        suggestion_text += f"   📍 {doctor['clinic']}, {doctor['province']}\n"
                        suggestion_text += f"   💰 {doctor['price']}\n\n"
                    
                    return base_response + suggestion_text
        
        return base_response

    def get_suggestions(self, partial: str = "") -> List[str]:
        """Gợi ý câu hỏi thường gặp hoặc theo từ khóa của người dùng."""
        default_suggestions = [
            "Tôi bị đau đầu, chóng mặt có sao không?",
            "Làm sao để đặt lịch khám bác sĩ?",
            "Khám chuyên khoa tim mạch ở đâu uy tín?",
            "Triệu chứng cảm cúm và cách phòng ngừa?",
            "Tôi bị đau dạ dày cần kiêng ăn gì?"
        ]
        if not partial or not partial.strip():
            return default_suggestions[:4]

        partial_lower = partial.strip().lower()
        matched = []

        # Kiểm tra trùng với câu hỏi mặc định
        for item in default_suggestions:
            if partial_lower in item.lower():
                matched.append(item)

        # Kiểm tra chuyên khoa & bác sĩ từ dữ liệu web
        try:
            web_data = self.web_data_manager.get_web_data()
            specialties = web_data.get('specialties', [])
            for sp in specialties:
                name = sp.get('name', '')
                if name and (partial_lower in name.lower() or any(w in name.lower() for w in partial_lower.split())):
                    matched.append(f"Tôi muốn tư vấn về chuyên khoa {name}")
                    matched.append(f"Danh sách bác sĩ chuyên khoa {name}")
                    break

            doctors = web_data.get('doctors', [])
            for doc in doctors:
                name = doc.get('name', '')
                if name and partial_lower in name.lower():
                    matched.append(f"Thông tin khám bệnh của {name}")
                    break
        except Exception as e:
            logger.warning(f"Error creating suggestions: {e}")

        # Bổ sung các gợi ý mặc định nếu chưa đủ
        for item in default_suggestions:
            if item not in matched:
                matched.append(item)
            if len(matched) >= 4:
                break

        return matched[:4]

    def summarize_medical_history(self, history_data: List[Dict]) -> str:
        """
        Tóm tắt lịch sử khám bệnh của bệnh nhân.
        """
        if not history_data:
            return "Không có dữ liệu lịch sử khám bệnh để tóm tắt."
        
        try:
            # Chuẩn bị nội dung lịch sử để tóm tắt
            history_text = ""
            for i, entry in enumerate(history_data, 1):
                raw_desc = entry.get("description", "{}")
                try:
                    details = json.loads(raw_desc)
                    desc_text = f"Chẩn đoán: {details.get('diagnosis', 'N/A')}. Triệu chứng: {details.get('clinicalExam', 'N/A')}. Điều trị: {details.get('treatment', 'N/A')}."
                except:
                    desc_text = raw_desc
                
                history_text += f"Lần {i}: Bác sĩ {entry.get('User', {}).get('firstName', '')} {entry.get('User', {}).get('lastName', '')}. {desc_text}\n"
            
            prompt = f"""Bạn là một chuyên gia y tế. Hãy tóm tắt lịch sử khám bệnh sau đây của bệnh nhân một cách cực kỳ ngắn gọn (giúp bác sĩ nắm bắt tình trạng trong 5 giây).
            Tập trung vào: các bệnh lý mãn tính, các triệu chứng lặp lại, và các chẩn đoán gần đây nhất.
            
            DỮ LIỆU LỊCH SỬ:
            {history_text}
            
            YÊU CẦU:
            1. Trả về kết quả bằng tiếng Việt.
            2. Sử dụng các gạch đầu dòng ngắn gọn.
            3. Không lặp lại các thông tin không quan trọng.
            4. Nếu có triệu chứng đáng lo ngại, hãy in đậm.
            """
            
            response = self.llm.invoke(prompt)
            if isinstance(response, dict) and "text" in response:
                return response["text"]
            return str(response)
            
        except Exception as e:
            logger.error(f"Error summarizing medical history: {e}")
            return "Xin lỗi, đã xảy ra lỗi khi tóm tắt bệnh án."

enhanced_chatbot = EnhancedMedicalChatbot()