import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def invert_image(image):
     return cv2.bitwise_not(image)
def thin_font (image, kernel_size, iterations = 1) :
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image = cv2.erode(image, kernel, iterations )
        return image
def fat_font (image, kernel_size, iterations = 1) :
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image = cv2.dilate(image, kernel, iterations )
        return image
def gray_image (image) :
        image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
        return image
def remove_noise (image, kernel_size, type ) :
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        """
        cv2.MORPH_OPEN	    :Erode rồi Dilate. Xóa điểm trắng nhỏ lẻ loi nhưng không ảnh hưởng nhiều đến hình chính.
        cv2.MORPH_CLOSE	    :Dilate rồi Erode. Lấp các lỗ nhỏ trong vật thể trắng hoặc nối các vùng trắng gần nhau.
        cv2.MORPH_GRADIENT	:Hiệu giữa Dilate và Erode. Cho ra đường viền của vật thể.
        cv2.MORPH_TOPHAT	:Hiệu giữa ảnh gốc và ảnh mở (original - open). Làm nổi bật chi tiết sáng nhỏ trên nền tối.
        cv2.MORPH_BLACKHAT	:Hiệu giữa ảnh đóng và ảnh gốc (close - original). Làm nổi bật chi tiết tối nhỏ trên nền sáng.
        """
        if type == 'open':
            image = cv2.bitwise_not (image)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = cv2.bitwise_not (image)
        elif type == 'close':
            image = cv2.bitwise_not (image)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.bitwise_not (image)
        elif type == 'gradient':
            image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        elif type == 'tophat':
            image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        elif type == 'blackhat':
            image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel) 
        else :
            raise ValueError("Invalid type. Choose from 'open', 'close', 'gradient', 'tophat', or 'blackhat'.")           
        # image = cv2.medianBlur(image, kernel_size)
        return image




