from flask import jsonify, request, Response
from core.chatbot import enhanced_chatbot
import json
import time
import logging

logger = logging.getLogger("enhanced_medical_chatbot")

def register_core_routes(app):
    @app.route('/load_model', methods=['POST'])
    def load_model():
        try:
            if not enhanced_chatbot.vector_store or not enhanced_chatbot.vector_store.vectorstore:
                enhanced_chatbot.refresh_data()
            return jsonify({
                "message": "Enhanced Medical Chatbot với LangChain đã sẵn sàng",
                "model": enhanced_chatbot.llm.model_name,
                "status": "loaded",
                "features": ["RAG", "PhoBERT Classification", "LangChain Integration"],
                "health": enhanced_chatbot.get_health_status()
            })
        except Exception as e:
            logger.error(f"Load model error: {e}")
            return jsonify({"error": "Failed to load model", "details": str(e)}), 500

    @app.route('/generate_stream_chatbox', methods=['POST'])
    def generate_stream_chatbox():
        data = request.json or {}
        question = data.get('text') or data.get('message') or data.get('question') or ''
        session_id = data.get('session_id') or data.get('sessionId') or 'default_session'
        location = data.get('location') or data.get('user_location') or data.get('province')
        if not question:
            return jsonify({"error": "Missing question"}), 400
        
        def generate():
            try:
                yield f"data: {json.dumps({'type': 'start', 'session_id': session_id}, ensure_ascii=False)}\n\n"
                for content in enhanced_chatbot.generate_streaming_response(question, session_id=session_id, location=location):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}\n\n"
                    time.sleep(0.04)
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_data = {'type': 'error', 'content': 'Lỗi khi streaming response'}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype='text/plain', headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        })

    @app.route('/generate_stream_question', methods=['POST'])
    def generate_stream_question():
        data = request.json or {}
        prompt = data.get('prompt') or data.get('text') or data.get('message') or data.get('question') or ''
        session_id = data.get('session_id') or data.get('sessionId') or 'default_session'
        location = data.get('location') or data.get('user_location') or data.get('province')
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400
        try:
            response = enhanced_chatbot.generate_response(prompt, session_id=session_id, location=location)
            return Response(
                json.dumps(response, ensure_ascii=False),  # ✅ giữ nguyên tiếng Việt
                content_type="application/json; charset=utf-8"
            )
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            return jsonify({"status": "error", "message": "Error processing request"}), 500

    @app.route('/suggestions', methods=['POST'])
    def suggest():
        data = request.json or {}
        partial = data.get('text', '')
        try:
            suggestions = enhanced_chatbot.get_suggestions(partial)
            return jsonify({"suggestions": suggestions})
        except Exception as e:
            logger.error(f"Suggestions error: {e}")
            return jsonify({"suggestions": [
                "Tôi bị đau đầu, có sao không?",
                "Khám sức khỏe ở đâu?",
                "Triệu chứng cảm cúm là gì?"
            ]}), 200

    @app.route('/clear_history', methods=['POST'])
    def clear():
        try:
            data = request.json or {}
            session_id = data.get('session_id') or data.get('sessionId')
            enhanced_chatbot.clear_memory(session_id=session_id)
            return jsonify({"message": "Đã xóa lịch sử hội thoại", "session_id": session_id or "default_session"})
        except Exception as e:
            logger.error(f"Clear history error: {e}")
            return jsonify({"message": "Lỗi khi xóa lịch sử"}), 500

    @app.route('/api/webhook/sync_data', methods=['POST'])
    @app.route('/refresh_data', methods=['POST'])
    def sync_data():
        try:
            enhanced_chatbot.refresh_data()
            return jsonify({
                "status": "success",
                "message": "Đã làm mới dữ liệu hệ thống thành công",
                "health": enhanced_chatbot.get_health_status()
            })
        except Exception as e:
            logger.error(f"Sync data error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    @app.route('/summarize_history', methods=['POST'])
    def summarize_history():
        data = request.json
        history = data.get('history', [])
        if not history:
            return jsonify({"error": "No history data provided"}), 400
        try:
            summary = enhanced_chatbot.summarize_medical_history(history)
            return jsonify({"summary": summary})
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return jsonify({"error": "Failed to generate summary", "details": str(e)}), 500
