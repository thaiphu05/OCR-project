import os

import google.generativeai as genai
from dotenv import load_dotenv

from models.model import Model


class LLMPostProcessor(Model):
    def __init__(self):
        load_dotenv()
        API_KEY = os.getenv(
            "API_KEY",
        )
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def load_model(self, model_path):
        return super().load_model(model_path)

    def predict(self, ocr_text):
        prompt = f"""
        Bạn là một trình sửa kết quả OCR. Dưới đây là kết quả OCR:

        "{ocr_text}"

        Hãy trả về văn bản chính xác, có nghĩa, chuẩn tiếng Việt.
        Không giải thích thêm. Chỉ trả về kết quả.
        """
        try:
            response = self.model.generate_content(prompt)
            corrected_text = response.text
            return corrected_text

        except Exception as e:
            print("LLM Postprocessing error:", e)
            return ocr_text
