"""segmentation.py"""

import numpy as np
import cv2
from skimage import filters, morphology, measure
from pontifficate.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def create_mask(
    image: np.ndarray, thresholding_method: str = "otsu", min_area: int = 100
) -> np.ndarray:
    """
    Create a binary mask for cell segmentation from an input image.

    Args:
        image (np.ndarray): Input grayscale image.
        thresholding_method (str): Method for thresholding ('otsu' or 'adaptive').
        min_area (int): Minimum area for objects to retain in the mask.

    Returns:
        np.ndarray: Binary mask with segmented cells.
    """
    logger.info("Starting mask creation...")

    if thresholding_method == "otsu":
        logger.info("Applying Otsu's thresholding...")
        threshold = filters.threshold_otsu(image)
        binary_mask = image > threshold
    elif thresholding_method == "adaptive":
        logger.info("Applying adaptive thresholding...")
        binary_mask = (
            cv2.adaptiveThreshold(
                image.astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
            > 0
        )
    else:
        logger.error(f"Unknown thresholding method: {thresholding_method}")
        raise ValueError(f"Unsupported thresholding method: {thresholding_method}")

    logger.info("Thresholding complete.")

    logger.info(f"Removing objects smaller than {min_area} pixels...")
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)

    logger.info("Mask creation complete.")
    return cleaned_mask


def segment_cells(mask: np.ndarray) -> list:
    """
    Label and segment cells from a binary mask.

    Args:
        mask (np.ndarray): Binary mask with segmented regions.

    Returns:
        list: List of labeled cell regions.
    """
    logger.info("Starting cell segmentation...")

    labeled_mask = measure.label(mask)
    logger.info(f"Found {labeled_mask.max()} connected components.")
    regions = measure.regionprops(labeled_mask)

    logger.info("Cell segmentation complete.")
    return regions


def draw_boundaries(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay cell boundaries on the original image.

    Args:
        image (np.ndarray): Original grayscale image.
        mask (np.ndarray): Binary mask with segmented cells.

    Returns:
        np.ndarray: Image with cell boundaries drawn.
    """
    logger.info("Drawing boundaries on the image...")

    boundaries = morphology.dilation(mask) ^ mask

    boundary_image = image.copy()
    boundary_image[boundaries] = boundary_image.max()

    logger.info("Boundary overlay complete.")
    return boundary_image
