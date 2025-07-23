from models.model import Model
from paddleocr import PaddleOCR
import cv2
class InferPaddleOCR(Model):
    def __init__ (self):
        self.ocr = PaddleOCR(
            # lang='vi',
            # use_angle_cls=True,
            cls_model_dir='C:/Users/admin/.paddlex/official_models/PP-LCNet_x1_0_textline_ori',
            det_model_dir='C:/Users/admin/.paddlex/official_models/PP-OCRv5_server_det',
            rec_model_dir='C:/Users/admin/.paddlex/official_models/PP-OCRv5_server_rec'
            # rec_model_dir='C:/Users/admin/.paddlex/official_models/latin_PP-OCRv5_mobile_rec'
        )

    def load_model(self, model_path):
        return super().load_model(model_path)

    def predict(self, image_path, bounding_box=False):
        if bounding_box ==False :
            try : 
                result = self.ocr.predict(image_path)
            except Exception as e:
                print("OCR error:", e)
                return []
            texts = []
            for item in result:
                if 'rec_texts' in item:
                    texts = item['rec_texts']
            return texts
        else:
            try:
                output = self.ocr.predict(image_path)
            except Exception as e:
                print("OCR error:", e)
                return []
            bounding_boxes = []
            texts = [] 
            for item in output:
                if 'rec_polys' in item:
                    bounding_boxes = item['rec_polys']
                if 'rec_texts' in item:
                    texts = item['rec_texts']
                    break
            result = []
            for box, text in zip(bounding_boxes, texts):
                result.append({
                    'box': box.tolist(),
                    'text': text
                })
            return result