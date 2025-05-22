#!/usr/bin/env python3
"""
Test script for paired DAPI/FITC analysis.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from pontifficate.paired_analysis import process_directory
from pontifficate.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)
logger.setLevel(logging.INFO)

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def main():
    """Run the paired analysis test on sample data with configurable segmentation options."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test paired DAPI/FITC analysis with configurable segmentation options")
    parser.add_argument("--no-watershed", action="store_true", help="Disable watershed segmentation")
    parser.add_argument("--no-edge-detection", action="store_true", help="Disable edge detection")
    parser.add_argument("--min-cell-area", type=int, default=200, help="Minimum cell area in pixels")
    parser.add_argument("--combined-csv", action="store_true", help="Generate a combined CSV report")
    args = parser.parse_args()
    
    # Get segmentation options from arguments
    use_watershed = not args.no_watershed
    use_edge_detection = not args.no_edge_detection
    min_cell_area = args.min_cell_area
    
    # Get project root path
    project_root = get_project_root()
    
    # Get the directory of the sample data
    data_dir = os.path.join(project_root, "data", "testing")
    
    # Create an output directory
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set path for combined CSV output if requested
    combined_csv_path = os.path.join(output_dir, "combined_analysis_results.csv") if args.combined_csv else None
    
    logger.info(f"Testing paired analysis with data from: {data_dir}")
    logger.info(f"Output will be saved to: {output_dir}")
    logger.info(f"Segmentation options: watershed={use_watershed}, edge_detection={use_edge_detection}, min_cell_area={min_cell_area}")
    if args.combined_csv:
        logger.info(f"Combined CSV will be saved to: {combined_csv_path}")
    
    # Process the directory
    try:
        results = process_directory(
            data_dir, 
            output_dir, 
            combined_csv_path,
            use_watershed=use_watershed,
            use_edge_detection=use_edge_detection,
            min_cell_area=min_cell_area
        )
        
        logger.info(f"Analysis complete. Processed {len(results)} image pairs.")
        
        # For each sample
        for result in results:
            sample_name = result['sample_name']
            cell_count = result['cell_count']
            logger.info(f"Sample: {sample_name}, Total cells: {cell_count}")
            
            # Output individual cell measurements
            if result['cell_measurements']:
                logger.info(f"  Individual cell measurements for {sample_name}:")
                logger.info(f"  {'Cell ID':<8} {'Foci Count':<12} {'Norm. Intensity':<18} {'Cell Area':<10}")
                logger.info(f"  {'-'*8} {'-'*12} {'-'*18} {'-'*10}")
                
                for cell in result['cell_measurements']:
                    cell_id = cell['cell_id']
                    foci_count = cell['foci_count']
                    norm_intensity = cell['normalized_intensity']
                    cell_area = cell['cell_area']
                    
                    logger.info(f"  {cell_id:<8} {foci_count:<12} {norm_intensity:<18.4f} {cell_area:<10}")
            else:
                logger.info(f"  No cells detected in {sample_name}")
            
            logger.info("")  # Empty line for better readability
            
            # Log the path to the individual CSV file
            csv_path = os.path.join(output_dir, f"{sample_name}_cell_analysis_results.csv")
            logger.info(f"Results for {sample_name} saved to: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 