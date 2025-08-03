import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from models.model import Model


class InferTableDetection(Model):
    def __init__(self):
        self.tabledetection = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-all"
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-all"
        )

    def load_model(self, model_path):
        return super().load_model(model_path)

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        target_sizes = [image.size[::-1]]
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": 400, "longest_edge": 1920},
        )
        with torch.no_grad():
            outputs = self.tabledetection(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )
        return results[0]
