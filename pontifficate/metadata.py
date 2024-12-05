"""metadata.py"""

import json

import numpy as np
import tifffile

from pontifficate.logging_config import setup_logger

logger = setup_logger(__name__)


def extract_metadata(file_path: str) -> dict:
    """
    Extract metadata from a TIFF file's ImageDescription tag.

    Args:
        file_path (str): Path to the TIFF file.

    Returns:
        dict: Parsed metadata as a Python dictionary.

    Raises:
        ValueError: If no ImageDescription tag is found in the TIFF file.
        RuntimeError: If an unexpected error occurs while extracting metadata.
    """
    try:
        logger.info(f"Extracting metadata from TIFF file: {file_path}")

        with tifffile.TiffFile(file_path) as tif:
            tags = tif.pages[0].tags
            logger.debug(tags)
            metadata = tags[34665].value
            logger.debug("bits per sample = tags[258].value")
            metadata["bitsPerSample"] = tags[258].value
            logger.debug(metadata)
            logger.info("Successfully extracted metadata...")
            return metadata

    except Exception as e:
        logger.error(f"Failed to extract metadata from TIFF file: {file_path} - {e}")
        raise RuntimeError(
            f"Failed to extract metadata from TIFF file: {file_path}"
        ) from e


def parse_metadata(metadata: dict) -> dict:
    """
    Parse and process relevant fields from the extracted metadata.

    Args:
        metadata (dict): Metadata dictionary extracted from a TIFF file.

    Returns:
        dict: Parsed and flattened metadata with relevant fields.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    try:
        logger.info("Parsing metadata...")

        # Extract and parse the UserComment field
        user_comment = metadata.get("UserComment", None)
        if user_comment is None:
            logger.warning("UserComment field is missing in metadata.")
            raise ValueError("UserComment field is missing in metadata.")

        try:
            parsed_comment = json.loads(user_comment)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse UserComment: {e}")
            raise ValueError("UserComment field is not valid JSON.")

        image_creation_summary = parsed_comment.get("imageCreationSummary", {})
        flattened_metadata = {
            "bitsPerSample": metadata.get("bitsPerSample"),
            "microscopeMode": parsed_comment.get("microscopeMode"),
            "imageCode": parsed_comment.get("imageCode"),
            "effectivePixelSize": parsed_comment.get("effectivePixelSize"),
            "objectiveMag": parsed_comment.get("objectiveMag"),
            "ledIntensity": image_creation_summary.get("ledIntensity"),
            "levelLow": image_creation_summary.get("levelLow"),
            "gain": image_creation_summary.get("gain"),
            "levelHigh": image_creation_summary.get("levelHigh"),
            "physicalChannelName": image_creation_summary.get("physicalChannelName"),
            "exposure": image_creation_summary.get("exposure"),
            "dhrIntensity": image_creation_summary.get("dhr", {}).get("intensity"),
            "dhrVersion": image_creation_summary.get("dhr", {}).get("version"),
            "levelMid": image_creation_summary.get("levelMid"),
        }

        logger.info(f"Successfully parsed metadata: {flattened_metadata}")
        return flattened_metadata

    except Exception as e:
        logger.error(f"Failed to parse metadata: {e}")
        raise ValueError("Failed to parse metadata.") from e
