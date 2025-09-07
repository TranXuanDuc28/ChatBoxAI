from flask import jsonify
from core.chatbot import enhanced_chatbot
import logging
import torch
import transformers
import langchain

logger = logging.getLogger("enhanced_medical_chatbot")

def register_management_routes(app):
    @app.route('/refresh_data', methods=['POST'])
    def refresh_data():
        try:
            enhanced_chatbot.refresh_data()
            return jsonify({
                "message": "Đã làm mới dữ liệu hệ thống",
                "status": "success",
                "health": enhanced_chatbot.get_health_status()
            })
        except Exception as e:
            logger.error(f"Refresh data error: {e}")
            return jsonify({"error": "Failed to refresh data", "details": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        try:
            health_status = enhanced_chatbot.get_health_status()
            return jsonify(health_status)
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 500

    @app.route('/system_info', methods=['GET'])
    def system_info():
        try:
            return jsonify({
                "system": "Enhanced Medical Chatbot with LangChain",
                "version": "2.0",
                "components": {
                    "llm": "Google Gemini 1.5 Flash",
                    "classifier": "PhoBERT Fine-tuned",
                    "embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
                    "vector_store": "FAISS",
                    "framework": "LangChain"
                },
                "dependencies": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "langchain": langchain.__version__,
                    "python": "3.8+"
                },
                "features": [
                    "Medical Question Classification",
                    "RAG (Retrieval-Augmented Generation)",
                    "Conversation Memory",
                    "Real-time Data Fetching",
                    "Streaming Responses",
                    "Multi-source Context"
                ],
                "health": enhanced_chatbot.get_health_status()
            })
        except Exception as e:
            logger.error(f"System info error: {e}")
            return jsonify({"error": str(e)}), 500