# exercises/contour_detection.py
"""
练习：轮廓检测

描述：
使用 OpenCV 检测图像中的轮廓并将其绘制出来。

请补全下面的函数 `contour_detection`。
"""
import cv2
import numpy as np

def contour_detection(image_path):
    """
    使用 OpenCV 检测图像中的轮廓
    参数:
        image_path: 图像路径
    返回:
        tuple: (绘制轮廓的图像, 轮廓列表) 或 (None, None) 失败时
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # OpenCV 4.x 返回两个值，3.x 返回三个值
        contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_info) == 3:
            _, contours, _ = contours_info
        else:
            contours, _ = contours_info
        contours = list(contours)
        img_draw = img.copy()
        cv2.drawContours(img_draw, contours, -1, (0, 255, 0), 2)
        return img_draw, contours
    except Exception:
        return None, None 