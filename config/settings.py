from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging

load_dotenv()

logger = logging.getLogger("enhanced_medical_chatbot")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BACKEND_API_BASE = os.getenv("BACKEND_API_BASE")
BACKEND_LANG = os.getenv("BACKEND_LANG")

os.environ["HF_HOME"] = os.getenv("HF_HOME")
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE")
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING")
os.environ["USE_TF"] = os.getenv("USE_TF")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set!")
    raise ValueError("Please set GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)