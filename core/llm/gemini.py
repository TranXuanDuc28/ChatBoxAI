import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import logging

logger = logging.getLogger("enhanced_medical_chatbot")

class GeminiLLM(LLM):
    # Khai báo field cho pydantic
    model_name: str = Field(default="gemini-1.5-flash", description="Tên model Gemini")

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
                    max_output_tokens=400,
                    temperature=0.7,
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."


class GeminiStreamingLLM(GeminiLLM):
    def stream_response(self, prompt: str):
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.7,
                ),
                stream=True,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            for chunk in response:
                try:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
                    elif hasattr(chunk, 'parts') and chunk.parts:
                        for part in chunk.parts:
                            if hasattr(part, 'text') and part.text:
                                yield part.text
                except Exception as chunk_error:
                    logger.error(f"Chunk processing error: {chunk_error}")
                    continue
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield "Xin lỗi, tôi đang gặp sự cố. Vui lòng thử lại sau."
