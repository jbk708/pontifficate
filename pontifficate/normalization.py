"""normalization.py"""

import numpy as np
from skimage import exposure

from pontifficate.logging_config import setup_logger

logger = setup_logger(__name__)


def rescale_intensity(
    image: np.ndarray, level_low: float, level_high: float, bits_per_sample: int = 16
) -> np.ndarray:
    """
    Rescale the intensity of an image based on metadata levels.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        level_low (float): Minimum intensity level (e.g., metadata levelLow).
        level_high (float): Maximum intensity level (e.g., metadata levelHigh).
        bits_per_sample (int): Number of bits per sample (e.g., 8 for 8-bit images).

    Returns:
        np.ndarray: Image with rescaled intensity.
    """
    logger.info("Rescaling image intensity...")
    if level_low is None or level_high is None:
        logger.warning("levelLow or levelHigh is None. Skipping intensity rescaling.")
        return image

    max_pixel_value = 2**bits_per_sample - 1
    in_range = (level_low * max_pixel_value, level_high * max_pixel_value)
    logger.info(f"Input range for rescaling: {in_range}")

    rescaled_image = exposure.rescale_intensity(
        image, in_range=in_range, out_range=(0, 65535)
    )
    logger.info("Intensity rescaling complete.")
    return rescaled_image


def normalize_background(image: np.ndarray) -> np.ndarray:
    """
    Perform background normalization on an image.

    Args:
        image (np.ndarray): Input image as a NumPy array.

    Returns:
        np.ndarray: Image with normalized background.
    """
    logger.info("Performing background normalization...")
    normalized_image = image - np.min(image)
    normalized_image = (
        normalized_image / np.max(normalized_image) * 65535
    )  
    logger.info("Background normalization complete.")
    return normalized_image.astype(np.uint16)

