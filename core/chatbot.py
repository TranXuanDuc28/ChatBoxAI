from core.llm.gemini import GeminiLLM, GeminiStreamingLLM
from core.classifier.medical_classifier import MedicalClassifier
from core.vector_store.enhanced_vector_store import EnhancedVectorStore
from core.vector_store.embeddings import SentenceTransformerEmbeddings
from core.data.web_data_manager import WebDataManager
from config.prompts import MEDICAL_SYSTEM_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import logging
from config.settings import GEMINI_API_KEY
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger("enhanced_medical_chatbot")

class EnhancedMedicalChatbot:
    def __init__(self):
        self.medical_classifier = MedicalClassifier()
        self.web_data_manager = WebDataManager()
        self.embeddings = SentenceTransformerEmbeddings()
        self.vector_store = EnhancedVectorStore(self.embeddings)
        self.llm = GeminiLLM()
        self.streaming_llm = GeminiStreamingLLM()
        self.memory = ConversationBufferWindowMemory(
            k=6,
            memory_key="chat_history",
            output_key="answer",  # Thêm output_key để chỉ định rõ key đầu ra
            return_messages=True
        )
        self.rag_chain = None
        self.suggestion_chain = None
        self.setup_chains()
    
    def setup_chains(self):
        web_data = self.web_data_manager.get_web_data()
        self.vector_store.build_from_web_data(web_data)
        
        if self.vector_store.vectorstore is not None:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
            prompt_template = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=MEDICAL_SYSTEM_PROMPT
            )
            self.rag_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template},
                output_key="answer"  # Đảm bảo output_key được đặt đúng
            )
            logger.info("RAG chain initialized successfully")
        else:
            logger.warning("Vector store not available, using simple LLM chain")
            prompt_template = PromptTemplate(
                input_variables=["chat_history", "question"],
                template="""Bạn là trợ lý y tế. 
                Chat history: {chat_history}
                Human: {question}
                Assistant:"""
            )
            self.rag_chain = prompt_template | self.llm
        
        suggestion_prompt = PromptTemplate(
            input_variables=["original_question"],
            template="""Người dùng vừa hỏi: "{original_question}"
            Tuy nhiên câu hỏi này không thuộc lĩnh vực y tế.
            Hãy đề xuất 1 câu hỏi tương tự nhưng liên quan đến y tế.
            Trả lời bằng 1 câu hỏi duy nhất, tiếng Việt:"""
        )
        self.suggestion_chain = suggestion_prompt | self.llm
    
    def refresh_data(self):
        logger.info("Refreshing web data...")
        self.web_data_manager.last_update = 0
        self.setup_chains()
    
    def get_health_status(self):
        """Get system health status"""
        return {
            "status": "healthy",
            "model": "gemini-1.5-flash",
            "classifier_loaded": self.medical_classifier.model is not None,
            "vector_store_ready": self.vector_store.vectorstore is not None,
            "rag_chain_ready": self.rag_chain is not None,
            "memory_messages": len(self.memory.chat_memory.messages),
            "api_key_configured": bool(GEMINI_API_KEY)
        }
    def get_smart_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh thông minh dựa trên câu hỏi của người dùng."""
        if self.vector_store.vectorstore is None:
            return "Không có thông tin bổ sung"
        
        try:
            # Phân tích ý định câu hỏi
            message_lower = user_message.lower()
            
            # Nếu hỏi về bác sĩ cụ thể
            if any(word in message_lower for word in ["bác sĩ", "doctor", "bs"]):
                return self._get_doctor_context(user_message)
            
            # Nếu hỏi về chuyên khoa
            elif any(word in message_lower for word in ["chuyên khoa", "khoa", "specialty"]):
                return self._get_specialty_context(user_message)
            
            # Nếu mô tả triệu chứng - tìm bác sĩ phù hợp
            elif any(word in message_lower for word in ["đau", "bệnh", "triệu chứng", "khám", "chữa"]):
                return self._get_symptom_context(user_message)
            
            # Tìm kiếm thông thường
            else:
                return self._get_general_context(user_message)
                
        except Exception as e:
            logger.warning(f"Smart context error: {e}")
            return self._get_general_context(user_message)

    def _get_doctor_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh về bác sĩ."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.get_relevant_documents(user_message)
        
        # Ưu tiên docs về bác sĩ và chuyên khoa liên quan
        doctor_docs = [doc for doc in docs if doc.metadata.get("type") in ["doctor", "specialty", "specialty_detail"]]
        other_docs = [doc for doc in docs if doc.metadata.get("type") not in ["doctor", "specialty", "specialty_detail"]]
        
        # Kết hợp ưu tiên doctor_docs
        combined_docs = doctor_docs[:10] + other_docs[:5]
        
        return "\n".join([doc.page_content for doc in combined_docs])

    def _get_specialty_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh về chuyên khoa."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.get_relevant_documents(user_message)
        
        # Ưu tiên docs về chuyên khoa và bác sĩ thuộc chuyên khoa đó
        specialty_docs = [doc for doc in docs if doc.metadata.get("type") in ["specialty", "specialty_detail", "doctor"]]
        other_docs = [doc for doc in docs if doc.metadata.get("type") not in ["specialty", "specialty_detail", "doctor"]]
        
        combined_docs = specialty_docs[:12] + other_docs[:3]
        
        return "\n".join([doc.page_content for doc in combined_docs])

    def _get_symptom_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh khi người dùng mô tả triệu chứng."""
        # Tìm kiếm rộng hơn để bao gồm cả bác sĩ và chuyên khoa
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        docs = retriever.get_relevant_documents(user_message)
        
        # Tìm thêm bác sĩ liên quan
        symptom_keywords = self._extract_symptom_keywords(user_message)
        if symptom_keywords:
            for keyword in symptom_keywords[:2]:  # Chỉ lấy 2 từ khóa chính
                extra_docs = retriever.get_relevant_documents(f"bác sĩ chuyên khoa {keyword}")
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
        """Trích xuất từ khóa triệu chứng để tìm chuyên khoa phù hợp."""
        symptom_mapping = {
            "tim": ["tim", "trái tim", "nhịp tim", "đau ngực"],
            "gan": ["gan", "mật", "vàng da", "đau bụng phải"],
            "thần kinh": ["đầu", "não", "thần kinh", "tê liệt", "chóng mặt"],
            "mắt": ["mắt", "mờ", "nhìn", "thị lực"],
            "tai mũi họng": ["tai", "mũi", "họng", "nghe", "nuốt"],
            "da": ["da", "ngứa", "nổi mẩn", "dị ứng"],
            "xương khớp": ["xương", "khớp", "đau lưng", "cột sống"],
            "nhi": ["trẻ em", "em bé", "con", "trẻ"],
            "sản phụ": ["mang thai", "sinh con", "phụ khoa", "kinh nguyệt"]
        }
        
        message_lower = message.lower()
        keywords = []
        
        for specialty, symptoms in symptom_mapping.items():
            for symptom in symptoms:
                if symptom in message_lower:
                    keywords.append(specialty)
                    break
        
        return list(set(keywords))

    def _get_general_context(self, user_message: str) -> str:
        """Lấy ngữ cảnh thông thường."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.get_relevant_documents(user_message)
        return "\n".join([doc.page_content for doc in docs])
    
    # Cập nhật phương thức generate_response để sử dụng smart context
    def generate_response(self, user_message: str) -> str:
        """
        Tạo phản hồi không streaming với ngữ cảnh thông minh và gợi ý bác sĩ.
        """
        try:
            # Lấy ngữ cảnh thông minh
            context = self.get_smart_context(user_message)
            
            # Lấy lịch sử trò chuyện
            chat_history = ""
            if self.memory.chat_memory.messages:
                recent_messages = self.memory.chat_memory.messages[-4:]
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        chat_history += f"{role}: {msg.content}\n"

            # Tạo prompt với MEDICAL_SYSTEM_PROMPT
            prompt = MEDICAL_SYSTEM_PROMPT.format(
                context=context,
                chat_history=chat_history,
                question=user_message
            )

            # Gọi LLM để tạo phản hồi
            response = self.llm.invoke(prompt)
            if isinstance(response, dict) and "text" in response:
                response = response["text"]

            # Tăng cường phản hồi với gợi ý bác sĩ
            enhanced_response = self.enhance_response_with_doctor_suggestions(user_message, response)

            # Cập nhật bộ nhớ trò chuyện
            try:
                self.memory.chat_memory.add_user_message(user_message)
                self.memory.chat_memory.add_ai_message(enhanced_response)
            except Exception as e:
                logger.warning(f"Memory update error: {e}")

            return enhanced_response
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."

    def generate_streaming_response(self, user_message: str):
        """
        Tạo phản hồi streaming với ngữ cảnh thông minh.
        """
        try:
            # Sử dụng smart context
            context = self.get_smart_context(user_message)
            
            # Lấy lịch sử trò chuyện
            chat_history = ""
            if self.memory.chat_memory.messages:
                recent_messages = self.memory.chat_memory.messages[-4:]
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        chat_history += f"{role}: {msg.content}\n"
            
            # Tạo prompt
            prompt = MEDICAL_SYSTEM_PROMPT.format(
                context=context,
                chat_history=chat_history,
                question=user_message
            )
            
            # Stream response từ LLM
            full_response = ""
            for chunk in self.streaming_llm.stream_response(prompt):
                full_response += chunk
                yield chunk
            
            # Sau khi stream xong, thêm gợi ý bác sĩ nếu cần
            doctor_suggestions = self._get_doctor_suggestions_for_streaming(user_message)
            if doctor_suggestions:
                yield "\n\n"  # Xuống dòng
                for suggestion_chunk in doctor_suggestions:
                    yield suggestion_chunk
                    full_response += suggestion_chunk
            
            # Cập nhật memory với response đầy đủ
            try:
                self.memory.chat_memory.add_user_message(user_message)
                self.memory.chat_memory.add_ai_message(full_response)
            except Exception as e:
                logger.warning(f"Memory update error: {e}")
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."

    def _get_doctor_suggestions_for_streaming(self, user_message: str) -> List[str]:
        """
        Tạo gợi ý bác sĩ dưới dạng chunks cho streaming.
        """
        try:
            message_lower = user_message.lower()
            suggestions = []
            
            # Nếu hỏi về triệu chứng
            if any(word in message_lower for word in ["đau", "bệnh", "triệu chứng", "khám"]):
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
                words = message_lower.split()
                for i, word in enumerate(words):
                    if word in ["chuyên", "khoa"] and i < len(words) - 1:
                        specialty_name = words[i + 1]
                        doctors = self.find_doctors_by_specialty(specialty_name, limit=3)
                        if doctors:
                            suggestions.append(f"👨‍⚕️ **Các bác sĩ chuyên khoa {specialty_name}:**\n")
                            for j, doctor in enumerate(doctors, 1):
                                doctor_info = f"{j}. **Bác sĩ {doctor['name']}**\n"
                                doctor_info += f"   📍 {doctor['clinic']}, {doctor['province']}\n"
                                doctor_info += f"   💰 {doctor['price']}\n\n"
                                suggestions.append(doctor_info)
                        break
            
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
        message_lower = user_message.lower()
        
        context_info = {
            "message": user_message,
            "detected_intent": "general",
            "keywords_found": [],
            "context_strategy": "general_search"
        }
        
        # Phân tích ý định
        if any(word in message_lower for word in ["bác sĩ", "doctor", "bs"]):
            context_info["detected_intent"] = "doctor_inquiry"
            context_info["context_strategy"] = "doctor_focused"
            context_info["keywords_found"] = [word for word in ["bác sĩ", "doctor", "bs"] if word in message_lower]
        
        elif any(word in message_lower for word in ["chuyên khoa", "khoa", "specialty"]):
            context_info["detected_intent"] = "specialty_inquiry"
            context_info["context_strategy"] = "specialty_focused"
            context_info["keywords_found"] = [word for word in ["chuyên khoa", "khoa", "specialty"] if word in message_lower]
        
        elif any(word in message_lower for word in ["đau", "bệnh", "triệu chứng", "khám", "chữa"]):
            context_info["detected_intent"] = "symptom_description"
            context_info["context_strategy"] = "symptom_to_doctor_matching"
            context_info["keywords_found"] = [word for word in ["đau", "bệnh", "triệu chứng", "khám", "chữa"] if word in message_lower]
            
            # Thêm symptom keywords
            symptom_keywords = self._extract_symptom_keywords(user_message)
            if symptom_keywords:
                context_info["symptom_keywords"] = symptom_keywords
        
        return context_info

    # Method để test hiệu suất của từng strategy
    def test_context_strategies(self, user_message: str) -> Dict:
        """
        Test các strategy khác nhau để so sánh hiệu quả.
        """
        if self.vector_store.vectorstore is None:
            return {"error": "Vector store not available"}
        
        strategies = {
            "general": self._get_general_context(user_message),
            "doctor_focused": self._get_doctor_context(user_message),
            "specialty_focused": self._get_specialty_context(user_message),
            "symptom_focused": self._get_symptom_context(user_message),
            "smart": self.get_smart_context(user_message)
        }
        
        # Đếm số lượng thông tin về bác sĩ/chuyên khoa trong mỗi strategy
        results = {}
        for strategy_name, context in strategies.items():
            doctor_mentions = context.lower().count("bác sĩ")
            specialty_mentions = context.lower().count("chuyên khoa")
            context_length = len(context)
            
            results[strategy_name] = {
                "context_length": context_length,
                "doctor_mentions": doctor_mentions,
                "specialty_mentions": specialty_mentions,
                "relevance_score": (doctor_mentions + specialty_mentions) / max(context_length / 1000, 1),
                "context_preview": context[:200] + "..." if len(context) > 200 else context
            }
        
        return results
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

    def suggest_doctors_for_symptoms(self, symptoms: str, limit: int = 3) -> List[Dict]:
        """
        Đề xuất bác sĩ dựa trên triệu chứng mô tả.
        """
        try:
            # Trích xuất keywords từ triệu chứng
            symptom_keywords = self._extract_symptom_keywords(symptoms)
            
            if not symptom_keywords:
                return []
            
            suggested_doctors = []
            
            # Tìm bác sĩ cho từng chuyên khoa liên quan
            for specialty in symptom_keywords:
                doctors = self.find_doctors_by_specialty(specialty, limit=2)
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
        
        # Nếu người dùng hỏi về triệu chứng
        if any(word in message_lower for word in ["đau", "bệnh", "triệu chứng", "khám"]):
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
            # Trích xuất tên chuyên khoa từ câu hỏi
            # Đây là logic đơn giản, có thể cải thiện bằng NER
            words = message_lower.split()
            for i, word in enumerate(words):
                if word in ["chuyên", "khoa"] and i < len(words) - 1:
                    specialty_name = words[i + 1]
                    doctors = self.find_doctors_by_specialty(specialty_name, limit=5)
                    if doctors:
                        suggestion_text = f"\n\n👨‍⚕️ **Các bác sĩ chuyên khoa {specialty_name}:**\n"
                        for i, doctor in enumerate(doctors, 1):
                            suggestion_text += f"{i}. **Bác sĩ {doctor['name']}**\n"
                            suggestion_text += f"   📍 {doctor['clinic']}, {doctor['province']}\n"
                            suggestion_text += f"   💰 {doctor['price']}\n\n"
                        
                        return base_response + suggestion_text
                    break
        
        return base_response

enhanced_chatbot = EnhancedMedicalChatbot()