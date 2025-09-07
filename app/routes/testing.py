from flask import jsonify, request
from core.chatbot import enhanced_chatbot
import logging
import json
import time

logger = logging.getLogger("enhanced_medical_chatbot")

def register_testing_routes(app):
    @app.route('/test_api', methods=['GET'])
    def test_api():
        try:
            test_response = enhanced_chatbot.llm._call("Xin chào, bạn có khỏe không?")
            return jsonify({
                "status": "success",
                "message": "Kết nối Gemini thành công",
                "response": test_response
            })
        except Exception as e:
            logger.error(f"API test error: {e}")
            return jsonify({
                "status": "error",
                "message": f"Lỗi kết nối Gemini: {str(e)}"
            }), 500

    @app.route('/test_classification', methods=['POST'])
    def test_classification():
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Missing text"}), 400
        try:
            is_medical, confidence = enhanced_chatbot.medical_classifier.is_medical_question(text)
            return jsonify({
                "text": text,
                "is_medical": is_medical,
                "confidence": confidence,
                "threshold": 0.5
            })
        except Exception as e:
            logger.error(f"Classification test error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/test_rag', methods=['POST'])
    def test_rag():
        data = request.json
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "Missing query"}), 400
        try:
            if enhanced_chatbot.vector_store.vectorstore is None:
                return jsonify({"error": "Vector store not initialized"}), 500
            retriever = enhanced_chatbot.vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query)
            results = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
            return jsonify({
                "query": query,
                "retrieved_documents": len(results),
                "results": results
            })
        except Exception as e:
            logger.error(f"RAG test error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/debug', methods=['POST'])
    def debug():
        data = request.json
        debug_type = data.get('type', 'full')
        try:
            debug_info = {
                "timestamp": time.time(),
                "debug_type": debug_type
            }
            if debug_type in ['full', 'memory']:
                debug_info["memory"] = {
                    "messages_count": len(enhanced_chatbot.memory.chat_memory.messages),
                    "messages": [
                        {
                            "type": msg.__class__.__name__,
                            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        }
                        for msg in enhanced_chatbot.memory.chat_memory.messages[-5:]
                    ]
                }
            if debug_type in ['full', 'vector_store']:
                debug_info["vector_store"] = {
                    "initialized": enhanced_chatbot.vector_store.vectorstore is not None,
                    "documents_count": len(enhanced_chatbot.vector_store.documents) if enhanced_chatbot.vector_store.documents else 0
                }
            if debug_type in ['full', 'web_data']:
                web_data = enhanced_chatbot.web_data_manager.get_web_data()
                debug_info["web_data"] = {
                    "clinics_count": len(web_data.get("clinics", [])),
                    "specialties_count": len(web_data.get("specialties", [])),
                    "doctors_count": len(web_data.get("doctors", [])),
                    "handbooks_count": len(web_data.get("handbooks", [])),
                    "last_update": enhanced_chatbot.web_data_manager.last_update,
                    "cache_duration": enhanced_chatbot.web_data_manager.cache_duration
                }
            return jsonify(debug_info)
        except Exception as e:
            logger.error(f"Debug error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/test_endpoints', methods=['GET'])
    def test_endpoints():
        endpoints = {
            "Core Endpoints": {
                "POST /load_model": "Initialize the chatbot",
                "POST /generate_stream_chatbox": "Chat with streaming response",
                "POST /generate_stream_question": "Q&A without streaming",
                "POST /suggestions": "Get question suggestions",
                "POST /clear_history": "Clear conversation history"
            },
            "Management Endpoints": {
                "POST /refresh_data": "Refresh web data and rebuild vector store",
                "GET /health": "System health check",
                "GET /system_info": "Detailed system information"
            },
            "Testing Endpoints": {
                "GET /test_api": "Test Gemini API connection",
                "POST /test_classification": "Test medical question classification",
                "POST /test_rag": "Test RAG document retrieval",
                "POST /debug": "Debug information (types: full, memory, vector_store, web_data)"
            }
        }
        return jsonify({
            "message": "Enhanced Medical Chatbot with LangChain - Available Endpoints",
            "server_status": "running",
            "model": "gemini-1.5-flash + langchain",
            "endpoints": endpoints
        })