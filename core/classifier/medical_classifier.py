import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_medical_chatbot")

class MedicalClassifier:
    def __init__(self, model_path: str = "TranXuanDuc28/my_phobert_model"):
        self.model_path = model_path
        self.labels = ["khong_y_te", "y_te"]
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            logger.info(f"Loading PhoBERT medical classifier from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            self.model.eval()
            logger.info("PhoBERT medical classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PhoBERT model: {e}")
            raise
    
    def is_medical_question(self, text: str, threshold: float = 0.5) -> Tuple[bool, float]:
        try:
            if not text or not isinstance(text, str):
                logger.error("Input text is invalid or empty")
                return False, 0.0
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                top_idx = torch.argmax(probs, dim=1).item()
                top_score = probs[0][top_idx].item()
                predicted_label = self.labels[top_idx]
            
            is_medical = predicted_label == "y_te" and top_score > threshold
            logger.info(f"Text: '{text[:50]}...' | Classification: {predicted_label} (confidence: {top_score:.3f})")
            return is_medical, top_score
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return False, 0.0

if __name__ == "__main__":
    classifier = MedicalClassifier()
    text = "Triệu chứng của viêm họng là gì?"
    is_medical, confidence = classifier.is_medical_question(text)
    print(f"Is medical: {is_medical}, Confidence: {confidence:.3f}")
