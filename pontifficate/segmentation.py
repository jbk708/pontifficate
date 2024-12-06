"""segmentation.py"""

import numpy as np
import cv2
from skimage import filters, morphology, measure
from skimage.filters import gaussian
from pontifficate.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def create_mask(
    image: np.ndarray, thresholding_method: str = "otsu", min_area: int = 1
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

    # Normalize or rescale the image

    logger.info("Rescaling uint16 image to uint8...")
    rescaled_image = (image / image.max() * 255).astype(np.uint8)

    # Apply thresholding
    if thresholding_method == "otsu":
        smoothed_image = gaussian(rescaled_image, sigma=2)
        logger.info("Applying Otsu's thresholding...")
        threshold = filters.threshold_otsu(smoothed_image)
        logger.info(f"Computed Otsu threshold: {threshold}")
        binary_mask = smoothed_image > threshold
    elif thresholding_method == "adaptive":
        logger.info("Applying adaptive thresholding...")
        binary_mask = (
            cv2.adaptiveThreshold(
                rescaled_image,
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

    # Remove small objects
    logger.info(f"Removing objects smaller than {min_area} pixels...")
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_area)

    logger.info("Mask creation complete.")
    return cleaned_mask


if __name__ == "__main__":
    from pontifficate.utils import read_tiff, save_tiff
    import tifffile
    import matplotlib.pyplot as plt
    import os

    # Load the uint16 image
    input_path = (
        "/home/jbk/side_projects/pontifficate/data/testing/Image_0001_tritc.tiff"
    )
    output_dir = "/home/jbk/side_projects/pontifficate/data/testing/"
    output_mask_path = os.path.join(output_dir, "mask.tiff")
    output_plot_path = os.path.join(output_dir, "mask_preview.png")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the input image
    image = read_tiff(input_path)

    # Create the mask
    mask = create_mask(image, thresholding_method="otsu")

    # Save the mask (rescale for saving)
    mask_uint8 = (mask * 255).astype(np.uint8)
    save_tiff(mask_uint8, output_mask_path)

    # Save the mask preview plot
    plt.figure()
    plt.imshow(mask_uint8, cmap="gray")
    plt.title("Mask Preview")
    plt.axis("off")
    plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)
    logger.info(f"Mask saved to: {output_mask_path}")
    logger.info(f"Mask preview plot saved to: {output_plot_path}")
