from models.infer_PaddleOCR import InferPaddleOCR
from utils.preprocessing import *
from utils.boundingbox import *
from utils.merge_box import *
from models.infer_TableDetection import InferTableDetection
import cv2
if __name__ == "__main__":
    ocr_model = InferPaddleOCR()
    table_model = InferTableDetection()
    image_path = 'data/test1.png'
    table_result = table_model.predict(image_path)
    texts = ocr_model.predict(image_path, bounding_box=True)
    normal_text = text_not_in_table(texts, table_result[0])
    merge = merge_table_ocr(table_result[0], texts)
    table = reconstruct_table(merge)
    for text in normal_text:
        print(text)
    print_table(table)
    # prompt_text = ''
    # for text in texts:
    #     print(text)
    #     prompt_text = prompt_text + ' ' + text
    # llm_post = LLMPostProcessor()
    # final_texts = llm_post.predict(prompt_text)
    # image_path = 'data/table.png'
    # for element in merge:
    #     print(element)
    # print(table_result)
