from models.model import Model
import os
from openai import OpenAI

class LLMPostProcessor(Model):
    def __init__(self, model="gpt-3.5-turbo"):
        # Khởi tạo client OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là một trợ lý AI giỏi sửa kết quả OCR tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            corrected_text = response.choices[0].message.content.strip()
            return corrected_text

        except Exception as e:
            print("LLM Postprocessing error:", e)
            return ocr_text
