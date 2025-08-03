from PIL import Image
from surya.layout import LayoutPredictor

from models.model import Model


class InferLayoutExtraction(Model):
    def __init__(self):
        self.layout_predictor = LayoutPredictor()

    def load_model(self, model_path):
        return super().load_model(model_path)

    def predict(self, image_path):
        image = Image.open(image_path)
        predictions = self.layout_predictor([image])
        return predictions[0]
