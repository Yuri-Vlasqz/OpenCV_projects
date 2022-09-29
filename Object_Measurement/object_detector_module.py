"""
Object Measurement Project: Contour detector module
"""


import cv2
import numpy as np


def pre_process_image(image: np.ndarray, denoise_type: str = 'GaussianBlur') -> np.ndarray:
    """
    Convert to grayscale and denoise image (apply before contour detection).

    Types of denoising:

    - GaussianBlur (default - fast)
    - medianBlur (fast)
    - bilateralFilter (medium)
    - fastNlMeans (slow)
    """

    # Convert Image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Denoising types ---
    if denoise_type == 'GaussianBlur':
        # Gaussian Blur Denoising (fast)
        denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0, 0)

    elif denoise_type == 'medianBlur':
        # Median Blur Denoising
        denoised_image = cv2.medianBlur(gray_image, 5)

    elif denoise_type == 'bilateralFilter':
        # Bilateral Filter Denoising (d > 5 slow)
        denoised_image = cv2.bilateralFilter(gray_image, 5, 75, 75)

    elif denoise_type == 'fastNlMeans':
        # Fast Nl Means Denoising (slowest)
        denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 20, 7, 21)

    else:
        print("Unknown denoise type")
        exit()

    return denoised_image


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Apply edge detection using the Canny algorithm with automatic upper and lower thresholds.
    """

    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower_tresh = int(max(0, (1.0 - sigma) * v))
    upper_tresh = int(min(255, (1.0 + sigma) * v))
    edge_mask = cv2.Canny(image, lower_tresh, upper_tresh, apertureSize=3, L2gradient=False)
    # print(lower, upper)

    return edge_mask


def detect_contours(image: np.ndarray, mask_type: str = 'adaptiveThreshold', test: bool = False) -> list:
    """
    Detect object edges/contours.

    Types of edge masking:

    - adaptiveThreshold (default - fast - more sensitive)
    - Canny (faster - less sentitive)
    """

    # --- Mask types ---

    if mask_type == 'adaptiveThreshold':
        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    elif mask_type == 'Canny':
        # Create a Mask with Canny Edge Detection
        # Manual thresholds:
        # mask = cv2.Canny(gray_dn, threshold1=90, threshold2=100, apertureSize=3, L2gradient=True)
        # Automatic thresholds:
        mask = auto_canny(image)

    else:
        print("Unknown mask type")
        exit()

    # --- Contour refinement ---

    # kernel size
    kernel = np.ones((5, 5), np.uint8)
    # Closing - Dilation followed by erosion (Good for removing noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Dilate here
    # mask = cv2.dilate(mask, kernel, iterations=1)

    # --- Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select contours
    objects_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500:
            # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
            objects_contours.append(cnt)

    # --- test info ---
    if test:
        # Show edge mask
        cv2.imshow("mask", mask)

    return objects_contours
