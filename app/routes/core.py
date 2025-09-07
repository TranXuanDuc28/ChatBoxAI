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
            enhanced_chatbot.refresh_data()
            return jsonify({
                "message": "Enhanced Medical Chatbot với LangChain đã sẵn sàng",
                "model": "gemini-1.5-flash",
                "status": "loaded",
                "features": ["RAG", "PhoBERT Classification", "LangChain Integration"],
                "health": enhanced_chatbot.get_health_status()
            })
        except Exception as e:
            logger.error(f"Load model error: {e}")
            return jsonify({"error": "Failed to load model", "details": str(e)}), 500

    @app.route('/generate_stream_chatbox', methods=['POST'])
    def generate_stream_chatbox():
        data = request.json
        question = data.get('text', '')
        if not question:
            return jsonify({"error": "Missing question"}), 400
        
        def generate():
            try:
                yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"
                for content in enhanced_chatbot.generate_streaming_response(question):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}\n\n"
                    time.sleep(0.05)
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
        data = request.json
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400
        try:
            response = enhanced_chatbot.generate_response(prompt)
            return jsonify({"response": response})
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            return jsonify({"status": "error", "message": "Error processing request"}), 500

    @app.route('/suggestions', methods=['POST'])
    def suggest():
        data = request.json
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
            enhanced_chatbot.clear_memory()
            return jsonify({"message": "Đã xóa lịch sử hội thoại"})
        except Exception as e:
            logger.error(f"Clear history error: {e}")
            return jsonify({"message": "Lỗi khi xóa lịch sử"}), 500
