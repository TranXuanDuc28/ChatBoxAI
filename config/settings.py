from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging

load_dotenv()

logger = logging.getLogger("enhanced_medical_chatbot")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BACKEND_API_BASE = os.getenv("BACKEND_API_BASE")
BACKEND_LANG = os.getenv("BACKEND_LANG")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set!")
    raise ValueError("Please set GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)