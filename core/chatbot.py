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
            return_messages=True
        )
        self.rag_chain = None
        self.suggestion_chain = None
        self.setup_chains()
    
    def setup_chains(self):
        web_data = self.web_data_manager.get_web_data()
        self.vector_store.build_from_web_data(web_data)
        
        if self.vector_store.vectorstore is not None:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 7})
            prompt_template = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template=MEDICAL_SYSTEM_PROMPT
            )
            self.rag_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt_template}
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
            # ✅ Dùng pipe thay LLMChain
            self.rag_chain = prompt_template | self.llm
        
        # ✅ Suggestion chain thay thế LLMChain
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
    
    def generate_response(self, user_message: str) -> str:
        try:
            if self.rag_chain is None:
                return "Hệ thống chưa được khởi tạo. Vui lòng thử lại."
            
            is_medical, confidence = self.medical_classifier.is_medical_question(user_message)
            if not is_medical and confidence > 0.7:
                try:
                    # ✅ invoke thay vì run
                    suggestion = self.suggestion_chain.invoke({"original_question": user_message})
                    # Nếu output là dict thì lấy key 'text'
                    if isinstance(suggestion, dict) and "text" in suggestion:
                        suggestion = suggestion["text"]
                    return f"❌ Tôi chỉ hỗ trợ trả lời các câu hỏi trong lĩnh vực **y tế**.\n\n💡 Gợi ý: {suggestion}"
                except Exception as e:
                    logger.error(f"Suggestion generation error: {e}")
                    return "❌ Tôi chỉ hỗ trợ trả lời các câu hỏi trong lĩnh vực **y tế**.\n\n💡 Hãy thử hỏi về sức khỏe, triệu chứng bệnh hoặc thuốc nhé."
            
            # ✅ chain có 2 loại: ConversationalRetrievalChain hoặc prompt|llm
            if hasattr(self.rag_chain, "invoke"):
                result = self.rag_chain.invoke({
                    "chat_history": self.memory.chat_memory.messages,
                    "question": user_message
                })
                if isinstance(result, dict) and "answer" in result:
                    return result["answer"]
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)
            else:
                return "Không thể tạo phản hồi từ chain."
                
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."
    def generate_streaming_response(self, user_message: str):
        try:
            is_medical, confidence = self.medical_classifier.is_medical_question(user_message)
            if not is_medical and confidence > 0.7:
                try:
                    suggestion = self.suggestion_chain.invoke({"original_question": user_message})
                    if isinstance(suggestion, dict) and "text" in suggestion:
                        suggestion = suggestion["text"]
                    error_msg = f"❌ Tôi chỉ hỗ trợ trả lời các câu hỏi trong lĩnh vực **y tế**.\n\n💡 Gợi ý: {suggestion}"
                except Exception:
                    error_msg = "❌ Tôi chỉ hỗ trợ trả lời các câu hỏi trong lĩnh vực **y tế**.\n\n💡 Hãy thử hỏi về sức khỏe, triệu chứng bệnh hoặc thuốc nhé."
                yield error_msg
                return
            
            # 🔹 Lấy context từ vectorstore
            context = ""
            if self.vector_store.vectorstore is not None:
                try:
                    retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.get_relevant_documents(user_message)
                   
                    context = "\n".join([doc.page_content for doc in docs])
                    print("context", context)
                except Exception as e:
                    logger.warning(f"Context retrieval error: {e}")
            
            # 🔹 Tạo chat_history text
            chat_history = ""
            if self.memory.chat_memory.messages:
                recent_messages = self.memory.chat_memory.messages[-4:]
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        chat_history += f"{role}: {msg.content}\n"
            
            # 🔹 Tạo prompt đầy đủ
            prompt = MEDICAL_SYSTEM_PROMPT.format(
                context=context if context else "Không có thông tin bổ sung",
                chat_history=chat_history,
                question=user_message
            )
            print("prompt", prompt)
            
            # 🔹 Stream response từ Gemini
            full_response = ""
            for chunk in self.streaming_llm.stream_response(prompt):
                full_response += chunk
                yield chunk
            
            # 🔹 Cập nhật memory
            try:
                self.memory.chat_memory.add_user_message(user_message)
                self.memory.chat_memory.add_ai_message(full_response)
            except Exception as e:
                logger.warning(f"Memory update error: {e}")
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."

enhanced_chatbot = EnhancedMedicalChatbot()