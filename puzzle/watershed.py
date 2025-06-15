import cv2
import numpy as np


def watershed_steps(image, kernel_size: int = 3, dist_thresh: float = 0.7):
    """Perform watershed segmentation and return intermediate images.

    Parameters
    ----------
    image : ndarray
        BGR input image.
    kernel_size : int, optional
        Size of the morphological kernel used for opening/dilation.
    dist_thresh : float, optional
        Threshold for the distance transform relative to its max value.

    Returns
    -------
    dict[str, ndarray]
        Dictionary of intermediate images with keys:
        ``gray``, ``thresh``, ``opening``, ``sure_bg``, ``sure_fg``,
        ``unknown``, ``markers``, ``result``.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image.copy(), markers)
    result = image.copy()
    result[markers == -1] = [255, 0, 0]

    return {
        "gray": gray,
        "thresh": thresh,
        "opening": opening,
        "sure_bg": sure_bg,
        "sure_fg": sure_fg,
        "unknown": unknown,
        "markers": markers.astype(np.int32),
        "result": result,
    }
