import random
import cv2


def draw_lines(image, lines, unique_colors=False):
    image_copy = image.copy()
    for line in range(len(lines)):
        color = (
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if unique_colors
            else (0, 255, 0)
        )

        x1, y1, x2, y2 = lines[line]

        cv2.line(image_copy, (x1, y1), (x2, y2), color, 4)
    return image_copy
