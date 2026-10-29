import requests
import json
import sys

CHATBOT_URL = "http://localhost:5002"

def print_section(title):
    print("\n" + "=" * 60)
    print(f"👉 {title}")
    print("=" * 60)

def test_health():
    print_section("1. Kiểm tra trạng thái hệ thống (Health Check)")
    try:
        res = requests.get(f"{CHATBOT_URL}/health")
        print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Không kết nối được tới Chatbot (chưa chạy server hoặc sai cổng): {e}")

def test_gemini_connection():
    print_section("2. Kiểm tra kết nối trực tiếp Gemini API")
    try:
        res = requests.get(f"{CHATBOT_URL}/test_api")
        print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Lỗi: {e}")

def test_rag_and_tool_calling(prompt):
    print_section(f"3. Thử nghiệm RAG & Function Calling với câu hỏi:\n   '{prompt}'")
    try:
        res = requests.post(
            f"{CHATBOT_URL}/generate_stream_question",
            json={"prompt": prompt}
        )
        print("\n🤖 Phản hồi từ AI Agent:")
        print(res.text)
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    # Đảm bảo in ký tự UTF-8 chính xác trên Windows Console
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)

    print("=== CHƯƠNG TRÌNH KIỂM TRA TỰ ĐỘNG CÁC TÍNH NĂNG NÂNG CAO ===")
    
    # 1. Test health
    test_health()
    
    # 2. Test Gemini API
    test_gemini_connection()
    
    # 3. Test RAG + Hybrid Search + Rerank
    test_rag_and_tool_calling("Chuyên khoa da liễu trị những bệnh gì?")
    
    # 4. Test AI Agent (Function Calling)
    test_rag_and_tool_calling("Bác sĩ Nguyễn Tiến Thành có lịch khám vào hôm nay thế nào?")
