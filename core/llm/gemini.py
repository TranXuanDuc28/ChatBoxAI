import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, List, Optional, Generator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import logging
import time
from config.settings import GEMINI_MODEL

logger = logging.getLogger("enhanced_medical_chatbot")

class GeminiLLM(LLM):
    # Khai báo field cho pydantic
    model_name: str = Field(default=GEMINI_MODEL, description="Tên model Gemini")

    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,  # ✅ TĂNG LÊN ĐỂ CHO PHÉP PHẢN HỒI DÀI
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                ),
                safety_settings=[
                    # ✅ GIẢM ĐỘ NGHIÊM NGẶT ĐỂ TRÁNH BLOCK PHẢN HỒI Y TẾ
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                ]
            )
            
            if hasattr(response, 'text') and response.text:
                return response.text
            else:
                logger.warning("No text in Gemini response")
                return "Xin lỗi, không thể tạo phản hồi. Vui lòng thử lại."
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            error_msg = str(e).lower()
            
            # ✅ XỬ LÝ CÁC LOẠI LỖI CỤ THỂ
            if "safety" in error_msg:
                return "Xin lỗi, nội dung phản hồi bị giới hạn do chính sách an toàn. Vui lòng hỏi câu khác."
            elif "quota" in error_msg or "rate" in error_msg:
                return "Hệ thống đang quá tải. Vui lòng thử lại sau ít phút."
            elif "timeout" in error_msg:
                return "Kết nối timeout. Vui lòng thử lại."
            else:
                return f"Xin lỗi, tôi đang gặp sự cố kỹ thuật: {str(e)}"


class GeminiStreamingLLM(GeminiLLM):
    def stream_response(self, prompt: str, max_retries: int = 3) -> Generator[str, None, None]:
        """
        Stream response với retry mechanism và error handling tốt hơn.
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting Gemini stream attempt {attempt + 1}/{max_retries}")
                
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=4000,  # ✅ TĂNG ĐỂ CHO PHÉP PHẢN HỒI DÀI
                        temperature=0.7,
                        top_p=0.8,
                        top_k=40,
                        # ✅ KHÔNG SET STOP SEQUENCES ĐỂ TRÁNH DỪNG SỚM
                    ),
                    stream=True,
                    safety_settings=[
                        # ✅ GIẢM ĐỘ NGHIÊM NGẶT
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                    ]
                )
                
                chunk_count = 0
                total_text = ""
                start_time = time.time()
                last_chunk_time = start_time
                
                # ✅ CẢI THIỆN CHUNK PROCESSING
                for chunk in response:
                    try:
                        current_time = time.time()
                        
                        # ✅ TIMEOUT PROTECTION
                        if current_time - last_chunk_time > 30:  # 30 giây không có chunk mới
                            logger.warning("No new chunks for 30 seconds, stopping")
                            if chunk_count == 0:
                                raise TimeoutError("No chunks received")
                            break
                        
                        chunk_text = ""
                        
                        # ✅ XỬ LÝ CHUNK LINH HOẠT HỠN
                        if hasattr(chunk, 'text') and chunk.text:
                            chunk_text = chunk.text
                        elif hasattr(chunk, 'parts') and chunk.parts:
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text:
                                    chunk_text += part.text
                        
                        if chunk_text:
                            chunk_count += 1
                            total_text += chunk_text
                            last_chunk_time = current_time
                            yield chunk_text
                        
                        # ✅ CHECK FINISH REASON
                        if hasattr(chunk, 'candidates'):
                            for candidate in chunk.candidates:
                                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                                    reason = candidate.finish_reason
                                    if hasattr(reason, 'name'):
                                        reason_name = reason.name
                                        logger.info(f"Stream finished with reason: {reason_name}")
                                        
                                        if reason_name == 'MAX_TOKENS':
                                            yield "\n\n📝 *[Phản hồi đã đạt giới hạn độ dài]*"
                                        elif reason_name in ['SAFETY', 'RECITATION']:
                                            yield f"\n\n⚠️ *[Phản hồi bị giới hạn: {reason_name}]*"
                                        
                                        # ✅ RETURN CHỈ KHI THỰC SỰ FINISHED
                                        if reason_name in ['STOP', 'MAX_TOKENS', 'SAFETY', 'RECITATION']:
                                            logger.info(f"Streaming completed: {chunk_count} chunks, {len(total_text)} chars")
                                            return
                        
                    except Exception as chunk_error:
                        logger.error(f"Chunk processing error: {chunk_error}")
                        continue
                
                # ✅ NẾU HẾT CHUNKS NHƯNG KHÔNG CÓ FINISH REASON
                if chunk_count > 0:
                    logger.info(f"Streaming completed naturally: {chunk_count} chunks, {len(total_text)} chars")
                    return
                else:
                    raise ValueError("No chunks received")
                    
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Streaming attempt {attempt + 1} failed: {e}")
                
                # ✅ NẾU LÀ LẦN THỬ CUỐI
                if attempt == max_retries - 1:
                    if "safety" in error_msg:
                        yield "⚠️ Phản hồi bị giới hạn do chính sách an toàn. Vui lòng thử câu hỏi khác."
                    elif "quota" in error_msg or "rate" in error_msg:
                        yield "⚠️ Hệ thống đang quá tải. Vui lòng thử lại sau ít phút."
                    elif "timeout" in error_msg:
                        yield "⚠️ Kết nối timeout. Vui lòng thử lại."
                    else:
                        yield f"❌ Lỗi kỹ thuật: {str(e)}"
                    return
                else:
                    # ✅ THỬ LẠI
                    yield f"🔄 Lỗi kết nối, đang thử lại lần {attempt + 2}...\n\n"
                    time.sleep(2)  # Đợi trước khi retry

    def test_connection(self) -> dict:
        """Test kết nối với Gemini API."""
        try:
            model = genai.GenerativeModel(self.model_name)
            test_response = model.generate_content(
                "Chào bạn!",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=50,
                    temperature=0.7,
                )
            )
            
            return {
                "status": "success",
                "model": self.model_name,
                "response": test_response.text if hasattr(test_response, 'text') else "No response",
                "response_length": len(test_response.text) if hasattr(test_response, 'text') else 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "model": self.model_name,
                "error": str(e)
            }