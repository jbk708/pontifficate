import logging
import os

import click
import numpy as np
import tifffile

from pontifficate.logging_config import setup_logger, set_global_log_level
from pontifficate.metadata import extract_metadata, parse_metadata
from pontifficate.normalization import normalize_background
from pontifficate.utils import save_tiff

# Initialize logger
logger = setup_logger(__name__)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def normalize(input_path, output_dir, log_level):
    """
    Normalize fluorescence microscopy images.

    INPUT_PATH: Path to a single image or a directory of images.
    OUTPUT_DIR: Directory to save normalized images.
    """
    # Set the global logger level
    numeric_log_level = getattr(logging, log_level.upper(), None)
    if numeric_log_level is not None:
        set_global_log_level(numeric_log_level)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine if input is a file or directory
    if os.path.isfile(input_path):
        # Single image normalization
        process_image(input_path, output_dir)
    elif os.path.isdir(input_path):
        # Batch processing
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(".tiff"):
                process_image(file_path, output_dir)


def process_image(file_path, output_dir):
    """
    Normalize a single image and save the output.

    Args:
        file_path (str): Path to the input image.
        output_dir (str): Path to the output directory.

    Returns:
        None
    """
    try:
        image = tifffile.imread(file_path)
        raw_metadata = extract_metadata(file_path)
        metadata = parse_metadata(raw_metadata)

        normalized_image = normalize_background(image)

        output_path = os.path.join(output_dir, os.path.basename(file_path))
        save_tiff(normalized_image, output_path, metadata)
        logger.info(f"Successfully processed: {file_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        click.echo(f"Failed to process {file_path}: {e}", err=True)
