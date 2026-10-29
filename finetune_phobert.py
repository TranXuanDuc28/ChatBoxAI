"""
MÃ NGUỒN FINE-TUNE PHOBERT PHÂN LOẠI Y TẾ
------------------------------------------
File này dùng để lưu trữ và chạy trên Google Colab / local có GPU để huấn luyện
mô hình phân loại ý định (Intent Classifier) cho Chatbot.

Yêu cầu cài đặt thư viện trước khi chạy:
pip install transformers[torch] datasets pandas scikit-learn huggingface_hub accelerate
"""

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch.nn.functional as F

def train_model():
    print("⏳ Bước 1: Đọc tập dữ liệu từ file CSV...")
    csv_path = "medical_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}. Đang tạo dữ liệu mẫu để chạy...")
        # Tạo dữ liệu giả lập nếu không có file CSV
        data = {
            "text": [
                "Tôi bị đau đầu và sốt cao từ hôm qua thì uống thuốc gì?",
                "Lịch khám của bác sĩ Nguyễn Văn A vào ngày mai thế nào?",
                "Triệu chứng của bệnh sốt xuất huyết là gì?",
                "Cho tôi hỏi giá khám chuyên khoa da liễu",
                "Chào bạn thời tiết hôm nay thế nào?",
                "Trận bóng đá tối nay mấy giờ chiếu thế?",
                "Làm thế nào để học giỏi lập trình Node.js?",
                "Tôi muốn đặt lịch khám nội tổng quát vào thứ hai tuần sau",
                "Cách làm món thịt kho tàu ngon tại nhà"
            ],
            "label": [1, 1, 1, 1, 0, 0, 0, 1, 0]
        }
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(csv_path)
    
    print(f"📊 Tổng số mẫu dữ liệu: {len(df)}")
    print(df.head())

    # Chia tập train/val (80% train, 20% validation)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Chuyển đổi DataFrame sang định dạng Dataset của Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("⏳ Bước 2: Tải Tokenizer của PhoBERT...")
    model_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Hàm tokenize văn bản với độ dài tối đa 256
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    print("⏳ Bước 3: Tải cấu hình mô hình PhoBERT pre-trained...")
    # Khởi tạo mô hình phân loại chuỗi với 2 nhãn đầu ra (y_te, khong_y_te)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Kiểm tra thiết bị chạy (Ưu tiên GPU CUDA nếu có)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"💻 Đang chạy huấn luyện trên thiết bị: {device.upper()}")

    print("⏳ Bước 4: Thiết lập các tham số huấn luyện...")
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,                  # Tốc độ học nhỏ cho fine-tuning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,                  # Huấn luyện trong 5 epochs
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        # Nếu muốn push trực tiếp lên Hugging Face Hub khi chạy trên Colab, bỏ comment các dòng dưới:
        # push_to_hub=True,
        # hub_model_id="my_phobert_model",
    )

    print("⏳ Bước 5: Khởi động quá trình Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
    )

    # Huấn luyện mô hình
    trainer.train()

    print("⏳ Bước 6: Lưu mô hình đã huấn luyện hoàn chỉnh...")
    output_model_dir = "./my_phobert_model_local"
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"🎉 Đã lưu mô hình và tokenizer thành công tại thư mục: '{output_model_dir}'")
    
    # Hướng dẫn đẩy lên Hugging Face Hub thủ công
    print("\n💡 Hướng dẫn upload lên Hugging Face Hub:")
    print("1. Chạy lệnh đăng nhập trong terminal hoặc notebook cell: huggingface-cli login")
    print("2. Sử dụng thư viện để push:")
    print("   model.push_to_hub('TranXuanDuc28/my_phobert_model')")
    print("   tokenizer.push_to_hub('TranXuanDuc28/my_phobert_model')")

if __name__ == "__main__":
    # Đăng nhập Hugging Face trước khi chạy nếu muốn push_to_hub=True
    # from huggingface_hub import login
    # login(token="YOUR_HF_WRITE_TOKEN")
    
    train_model()
