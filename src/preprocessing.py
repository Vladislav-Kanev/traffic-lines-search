import cv2
import numpy as np

from src.lines import get_center_lines, split_lines


def prepocess_image(image):
    height, width = image.shape[:2]
    gradient_mask = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        gradient_mask[y, :] = int(min(255 * (y / height) / 2, 255))

    gradient_image = cv2.merge([gradient_mask, gradient_mask, gradient_mask])

    blended_image = cv2.addWeighted(image, 0.5, gradient_image, 0.5, 0)

    gray = cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(blurred, (3, 3), 0)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

    return blurred


def get_road_marking(image):
    road_markings = cv2.HoughLinesP(
        image, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20
    )
    splitted_lines = split_lines(road_markings)
    center_lines = get_center_lines(splitted_lines)

    return center_lines
