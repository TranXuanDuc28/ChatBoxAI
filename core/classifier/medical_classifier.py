import os

class MedicalClassifier:
    def __init__(self, model_path: str = "TranXuanDuc28/my_phobert_model"):
        self.model_path = model_path
        self.labels = ["khong_y_te", "y_te"]
        self.tokenizer = None
        self.model = None

        # cache_dir trong Volume (giữ lại sau restart)
        self.cache_dir = "/mnt/data/model_cache"

        # Tạo thư mục nếu chưa có
        os.makedirs(self.cache_dir, exist_ok=True)

        self.load_model()
    
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import logging
        logger = logging.getLogger("enhanced_medical_chatbot")

        try:
            logger.info(f"Loading PhoBERT from {self.model_path} with cache at {self.cache_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self.model.eval()
            logger.info("PhoBERT medical classifier loaded successfully ✅")
        except Exception as e:
            logger.error(f"Error loading PhoBERT model: {e}")
            raise
