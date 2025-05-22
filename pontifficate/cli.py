import logging
import os

import click
import numpy as np
import tifffile

from pontifficate.logging_config import set_global_log_level, setup_logger
from pontifficate.metadata import extract_metadata, parse_metadata
from pontifficate.normalization import normalize_background, rescale_intensity
from pontifficate.utils import save_tiff
from pontifficate.paired_analysis import process_directory

# Initialize logger
logger = setup_logger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Set the logging level.",
)
def cli(log_level):
    """Command line interface for pontifficate."""
    # Set the global logger level
    numeric_log_level = getattr(logging, log_level.upper(), None)
    if numeric_log_level is not None:
        set_global_log_level(numeric_log_level)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def normalize(input_path, output_dir):
    """
    Normalize fluorescence microscopy images.

    INPUT_PATH: Path to a single image or a directory of images.
    OUTPUT_DIR: Directory to save normalized images.
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(input_path):
        process_image(input_path, output_dir)
    elif os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(".tiff"):
                process_image(file_path, output_dir)


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--combined-csv", "csv_path", type=click.Path(), help="Path to save combined CSV report (optional)")
@click.option("--watershed/--no-watershed", default=True, help="Use watershed segmentation to separate touching cells")
@click.option("--edge-detection/--no-edge-detection", default=True, help="Use edge detection to improve cell boundaries")
@click.option("--min-cell-area", default=200, type=int, help="Minimum area in pixels for a region to be considered a cell")
def analyze_pairs(input_dir, output_dir, csv_path, watershed, edge_detection, min_cell_area):
    """
    Analyze paired DAPI/FITC images.

    INPUT_DIR: Directory containing paired DAPI and FITC images.
    OUTPUT_DIR: Directory to save analysis results.
    
    Each image pair will produce:
    - A CSV file with cell measurements (sample_name_cell_analysis_results.csv)
    - A visualization of cell segmentation and foci (sample_name_visualization.png)
    - A visualization of segmentation steps (sample_name_segmentation_steps.png)
    """
    logger.info(f"Starting paired DAPI/FITC analysis on directory: {input_dir}")
    logger.info(f"Results will be saved to: {output_dir}")
    if csv_path:
        logger.info(f"Combined CSV report will be saved to: {csv_path}")
    logger.info(f"Segmentation options: watershed={watershed}, edge_detection={edge_detection}, min_cell_area={min_cell_area}")
    
    # Process the directory with the specified segmentation options
    results = process_directory(
        input_dir, 
        output_dir, 
        csv_path,
        use_watershed=watershed,
        use_edge_detection=edge_detection,
        min_cell_area=min_cell_area
    )
    
    logger.info(f"Analysis complete. Processed {len(results)} image pairs.")


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
        scaled_image = rescale_intensity(
            normalized_image,
            metadata["levelLow"],
            metadata["levelHigh"],
            metadata["bitsPerSample"],
        )
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        save_tiff(scaled_image, output_path, metadata)
        logger.info(f"Successfully processed: {file_path}")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        click.echo(f"Failed to process {file_path}: {e}", err=True)


if __name__ == "__main__":
    cli()
