"""
Общие утилиты для проекта.
Обработка Unicode-путей в OpenCV на Windows.
"""

import os
import cv2
import numpy as np


def cv2_imwrite_unicode(filepath, img, params=None):
    """cv2.imwrite, поддерживающий Unicode/кириллические пути на Windows."""
    ext = os.path.splitext(filepath)[1]
    if params:
        result, encoded = cv2.imencode(ext, img, params)
    else:
        result, encoded = cv2.imencode(ext, img)
    if result:
        encoded.tofile(filepath)
    return result


def cv2_imread_unicode(filepath, flags=cv2.IMREAD_COLOR):
    """cv2.imread, поддерживающий Unicode/кириллические пути на Windows."""
    data = np.fromfile(filepath, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img
