import cv2
import os
import numpy as np
from cv2.typing import MatLike

from .helpers import convert_to_bgr, convert_to_lab, get_blurred_gray
from .segment_params import SegmentParams


def segment_beans(image: np.ndarray, params: SegmentParams, debug: bool = True) -> list[np.ndarray]:
    """Return list of contours for each bean using connected components"""
    os.makedirs("steps", exist_ok=True)
    blur = _preprocess_to_gray(image)
    white_foreground = _binarize_to_foreground(blur)
    opened = _open_foreground(white_foreground, params.open_ksize)

    if debug:
        result_vis = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_vis, contours, -1, (0, 255, 0), 2)
        
        for filename, img in {
            "1_blur": blur,
            "2_white_foreground": white_foreground,
            "3_opened": opened,
            "4_result": result_vis,
        }.items():
            cv2.imwrite(f"steps/{filename}.png", img)

    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        c for c in contours 
        if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]

    return valid_contours


def segment_single_bean(image: np.ndarray, params: SegmentParams) -> list[np.ndarray]:
    """Segment single bean from training image using simple thresholding."""
    blur = _preprocess_to_gray(image)
    threshold = _binarize_to_foreground(blur)
    threshold = _open_foreground(threshold, params.open_ksize)

    # Find contours
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area and return the largest valid contour
    valid_contours = [
        c for c in contours if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]

    return valid_contours


def _normalize_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel in Lab for gentle contrast normalization."""
    lab = convert_to_lab(img_bgr)
    brightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_brightness = clahe.apply(brightness)
    lab_enhanced = cv2.merge([enhanced_brightness, green_red, blue_yellow])
    return convert_to_bgr(lab_enhanced)


def _preprocess_to_gray(image: np.ndarray) -> np.ndarray:
    image = _normalize_contrast(image)
    return get_blurred_gray(image)


def _binarize_to_foreground(image: np.ndarray) -> np.ndarray:
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return threshold if threshold.mean() < 127 else cv2.bitwise_not(threshold)


def _open_foreground(
    image: np.ndarray,
    ksize: int,
) -> np.ndarray:
    k_shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, k_shape, iterations=1)


def get_contours(image: MatLike, single_bean: bool) -> list[np.ndarray]:
    if single_bean:
        return segment_single_bean(
            image,
            SegmentParams(
                min_area=1000,
                max_area=100000,
                open_ksize=5,
            ),
        )
    return segment_beans(
        image,
        SegmentParams(
            min_area=6000,
            max_area=25_000,
            open_ksize=7,
        ),
    )
