import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger("enhanced_medical_chatbot")

class MedicalClassifier:
    def __init__(self, model_path: str = "./models/phobert_yte_cls"):
        self.model_path = model_path
        self.labels = ["khong_y_te", "y_te"]
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            logger.info("Loading PhoBERT medical classifier...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            logger.info("PhoBERT medical classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PhoBERT: {e}")
            raise

    
    def is_medical_question(self, text: str, threshold: float = 0.5) -> tuple[bool, float]:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                top_idx = torch.argmax(probs, dim=1).item()
                top_score = probs[0][top_idx].item()
                predicted_label = self.labels[top_idx]
            
            is_medical = predicted_label == "y_te" and top_score > threshold
            logger.info(f"Classification: {predicted_label} (confidence: {top_score:.3f})")
            return is_medical, top_score
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return True, 0.5