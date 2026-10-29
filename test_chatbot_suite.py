import sys
import io
import requests
import json
import time

# Ensure UTF-8 output for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://localhost:5002"

def print_result(name, status, response):
    color = "\033[92m" if status == "PASS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{color}{status}{reset}] {name}")
    if status == "FAIL":
        print(f"    Error: {response}")
    print("-" * 50)

def test_health():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            print_result("Health Check", "PASS", resp.json())
        else:
            print_result("Health Check", "FAIL", f"Status {resp.status_code}")
    except Exception as e:
        print_result("Health Check", "FAIL", str(e))

def test_classification():
    cases = [
        {"text": "Tôi bị đau đầu và buồn nôn", "expected": True},
        {"text": "Hôm nay trời có mưa không?", "expected": False},
        {"text": "Tìm bác sĩ chuyên khoa tim mạch", "expected": True},
        {"text": "Cổ phiếu VinFast hôm nay thế nào?", "expected": False}
    ]
    
    for case in cases:
        try:
            resp = requests.post(f"{BASE_URL}/test_classification", json={"text": case['text']})
            res = resp.json()
            is_med = res.get('is_medical')
            status = "PASS" if is_med == case['expected'] else "FAIL"
            print_result(f"Classification: '{case['text']}'", status, res)
        except Exception as e:
            print_result(f"Classification: '{case['text']}'", "FAIL", str(e))

def test_rag_retrieval():
    queries = [
        "bác sĩ chuyên khoa thần kinh",
        "phòng khám ở Hà Nội",
        "hướng dẫn khám bệnh"
    ]
    for q in queries:
        try:
            resp = requests.post(f"{BASE_URL}/test_rag", json={"query": q})
            res = resp.json()
            status = "PASS" if res.get('retrieved_documents', 0) > 0 else "FAIL"
            print_result(f"RAG Retrieval: '{q}'", status, res)
        except Exception as e:
            print_result(f"RAG Retrieval: '{q}'", "FAIL", str(e))

def test_medical_summary():
    dummy_history = [
        {"role": "user", "content": "Chào bác sĩ, tôi bị đau họng 3 ngày nay."},
        {"role": "assistant", "content": "Chào bạn, ngoài đau họng bạn có sốt hay ho không?"},
        {"role": "user", "content": "Tôi có sốt nhẹ và ho có đờm."},
        {"role": "assistant", "content": "Dựa trên triệu chứng, bạn có thể bị viêm họng cấp. Bạn nên đi khám tai mũi họng."}
    ]
    try:
        resp = requests.post(f"{BASE_URL}/summarize_history", json={"history": dummy_history})
        res = resp.json()
        status = "PASS" if "summary" in res and res['summary'] else "FAIL"
        print_result("Medical Summarization", status, res)
    except Exception as e:
        print_result("Medical Summarization", "FAIL", str(e))

def test_chat_streaming():
    print("Testing Chat Streaming (Manual check needed for content)...")
    try:
        resp = requests.post(f"{BASE_URL}/generate_stream_chatbox", 
                            json={"text": "Tôi nên làm gì khi bị cảm cúm?"}, 
                            stream=True)
        chunk_count = 0
        for line in resp.iter_lines():
            if line:
                chunk_count += 1
        status = "PASS" if chunk_count > 2 else "FAIL"
        print_result("Chat Streaming", status, f"Received {chunk_count} chunks")
    except Exception as e:
        print_result("Chat Streaming", "FAIL", str(e))

if __name__ == "__main__":
    print("=" * 20 + " CHATBOT TEST SUITE " + "=" * 20)
    test_health()
    test_classification()
    test_rag_retrieval()
    test_medical_summary()
    test_chat_streaming()
    print("=" * 60)
