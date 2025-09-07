from flask import jsonify
import logging

logger = logging.getLogger("enhanced_medical_chatbot")

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Endpoint not found",
            "message": "Use GET /test_endpoints to see available endpoints"
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal server error",
            "message": "Please check server logs for details"
        }), 500