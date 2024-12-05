"""utils.py"""

import json

import numpy as np
import tifffile

from pontifficate.logging_config import setup_logger

logger = setup_logger(__name__)


def read_tiff(file_path: str) -> np.ndarray:
    """
    Read a TIFF image and return it as a NumPy array.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    try:
        logger.info(f"Attempting to read TIFF file: {file_path}")
        with tifffile.TiffFile(file_path) as tif:
            image = tif.asarray()
        logger.info(f"Successfully read TIFF file with shape {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Failed to read TIFF file: {file_path} - {e}")
        raise RuntimeError(f"Failed to read TIFF file: {file_path}") from e


def save_tiff(
    image: np.ndarray,
    file_path: str,
    metadata: dict = None,
    metadata_tag_id: int = 65000,
) -> None:
    """
    Save a normalized image as a TIFF file with metadata stored in a single tag.

    Args:
        image (np.ndarray): Image to save as a TIFF file.
        file_path (str): Destination file path for the TIFF file.
        metadata (dict, optional): Metadata to include in the TIFF file.
        metadata_tag_id (int, optional): Tag ID to store the metadata. Defaults to 65000.

    Returns:
        None
    """
    try:
        logger.info(f"Saving image to {file_path}...")

        # Serialize metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None
        extratags = []

        # Add metadata to a single tag if provided
        if metadata_json:
            extratags.append(
                (metadata_tag_id, "s", len(metadata_json), metadata_json, True)
            )

        # Save the TIFF file
        tifffile.imwrite(file_path, image.astype(np.uint16), extratags=extratags)
        logger.info(f"Image successfully saved to {file_path}.")

    except Exception as e:
        logger.error(f"Failed to save TIFF file to {file_path}: {e}")
        raise RuntimeError(f"Failed to save TIFF file to {file_path}.") from e
