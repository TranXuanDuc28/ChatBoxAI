import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app import create_app
from utils.logging import setup_logging
from core.chatbot import enhanced_chatbot
from config.settings import GEMINI_API_KEY, BACKEND_API_BASE, BACKEND_LANG

logger = setup_logging()

if __name__ == '__main__':
    app = create_app()
    
    print("🚀 Enhanced Medical Chatbot with LangChain")
    print("=" * 50)
    print("📦 Checking dependencies...")
    print("✅ Google Generative AI configured")
    print("✅ LangChain integrated")
    print("✅ PhoBERT classification model")
    print("✅ FAISS vector store")
    print("✅ Enhanced multi-session memory management")
    print("✅ Hybrid RAG pipeline")
    print("✅ CORS configured")
    
    print("\n🔍 System configuration...")
    print(f"✅ Gemini API key: {'✓ Configured' if GEMINI_API_KEY else '✗ Missing'}")
    print(f"✅ Backend API: {BACKEND_API_BASE}")
    print(f"✅ Language: {BACKEND_LANG}")
    
    try:
        health_status = enhanced_chatbot.get_health_status()
        print(f"✅ Classifier loaded: {'✓' if health_status.get('classifier_loaded') else '✗'}")
        print(f"✅ Vector store ready: {'✓' if health_status.get('vector_store_ready') else '✗'}")
        print(f"✅ Active sessions: {health_status.get('active_sessions', 0)}")
    except Exception as e:
        print(f"⚠️ Initialization warning: {e}")
    
    print("\n🚀 Starting server...")
    print("📍 Server will run at: http://localhost:5002")
    print("\n📋 Available endpoints:")
    print("   Core:")
    print("   - POST /load_model: Initialize chatbot")
    print("   - POST /generate_stream_chatbox: Streaming chat")
    print("   - POST /generate_stream_question: Q&A")
    print("   - POST /suggestions: Question suggestions")
    print("   - POST /clear_history: Clear conversation")
    print("   Management:")
    print("   - POST /refresh_data: Refresh web data")
    print("   - GET /health: Health check")
    print("   - GET /system_info: System information")
    print("   Testing:")
    print("   - GET /test_api: Test Gemini API")
    print("   - POST /test_classification: Test medical classification")
    print("   - POST /test_rag: Test RAG retrieval")
    print("   - POST /debug: Debug information")
    print("   - GET /test_endpoints: List all endpoints")
    
    print("\n🎯 New LangChain features:")
    print("   - Advanced conversation memory")
    print("   - Standardized RAG pipeline")
    print("   - Better error handling")
    print("   - Component modularity")
    print("   - Enhanced debugging")
    
    print("\n💡 Press Ctrl+C to stop server")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5002, debug=True)