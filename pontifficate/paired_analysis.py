"""paired_analysis.py"""

import os
import csv
import numpy as np
from typing import List, Dict, Tuple
from skimage import measure, filters
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pontifficate.logging_config import setup_logger
from pontifficate.utils import read_tiff, save_tiff
from pontifficate.segmentation import create_mask
from pontifficate.normalization import normalize_background, rescale_intensity
from pontifficate.metadata import extract_metadata, parse_metadata

# Initialize logger
logger = setup_logger(__name__)


def find_image_pairs(directory: str) -> List[Dict[str, str]]:
    """
    Find paired DAPI and FITC images in a directory.
    
    Args:
        directory (str): Path to directory containing image files
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with sample name, dapi path, and fitc path
    """
    pairs = {}
    
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.tiff'):
            continue
            
        if '_dapi.tiff' in filename.lower():
            sample_name = filename.lower().replace('_dapi.tiff', '')
            if sample_name not in pairs:
                pairs[sample_name] = {'sample': sample_name, 'dapi': None, 'fitc': None}
            pairs[sample_name]['dapi'] = os.path.join(directory, filename)
            
        elif '_fitc.tiff' in filename.lower():
            sample_name = filename.lower().replace('_fitc.tiff', '')
            if sample_name not in pairs:
                pairs[sample_name] = {'sample': sample_name, 'dapi': None, 'fitc': None}
            pairs[sample_name]['fitc'] = os.path.join(directory, filename)
    
    # Return only complete pairs
    complete_pairs = [pair for pair in pairs.values() if pair['dapi'] and pair['fitc']]
    logger.info(f"Found {len(complete_pairs)} complete DAPI/FITC image pairs")
    
    return complete_pairs


def detect_foci(image: np.ndarray, cell_mask: np.ndarray, min_size: int = 3) -> Tuple[int, np.ndarray]:
    """
    Detect and count foci within a cell mask that are significantly brighter than background.
    
    Args:
        image (np.ndarray): FITC image
        cell_mask (np.ndarray): Binary mask for a single cell
        min_size (int): Minimum size of foci in pixels
        
    Returns:
        Tuple[int, np.ndarray]: Number of foci and mask of detected foci
    """
    # Apply cell mask to the FITC image
    masked_image = image.copy()
    masked_image[~cell_mask] = 0
    
    if not np.any(cell_mask):
        return 0, np.zeros_like(cell_mask)
    
    # Calculate local threshold for bright spots (foci)
    # Use Otsu's method on the masked area to find bright spots
    masked_values = masked_image[cell_mask]
    if len(masked_values) < 3:  
        return 0, np.zeros_like(cell_mask)
    
    try:
        threshold = filters.threshold_otsu(masked_values)
        threshold = threshold * 1.3  # Adjust this factor based on testing
    except:
        logger.warning("Could not determine threshold for foci detection")
        return 0, np.zeros_like(cell_mask)
    
    foci_mask = np.zeros_like(cell_mask)
    foci_mask[cell_mask] = masked_image[cell_mask] > threshold
    
    labeled_foci = measure.label(foci_mask)
    regions = measure.regionprops(labeled_foci)
    
    valid_foci = np.zeros_like(foci_mask)
    count = 0
    
    for region in regions:
        if region.area >= min_size:
            valid_foci[labeled_foci == region.label] = 1
            count += 1
    
    return count, valid_foci


def analyze_image_pair(dapi_path: str, fitc_path: str, output_dir: str = None, 
                       use_watershed: bool = True, use_edge_detection: bool = True,
                       min_cell_area: int = 100) -> Dict:
    """
    Analyze a paired DAPI/FITC image set.
    
    Args:
        dapi_path (str): Path to DAPI image
        fitc_path (str): Path to FITC image
        output_dir (str, optional): Directory to save visualization outputs
        use_watershed (bool): Whether to use watershed segmentation for cell separation
        use_edge_detection (bool): Whether to use edge detection for improved boundaries
        min_cell_area (int): Minimum area in pixels for a region to be considered a cell
        
    Returns:
        Dict: Analysis results with sample name, cell measurements
    """
    sample_name = os.path.basename(dapi_path).replace('_dapi.tiff', '')
    logger.info(f"Analyzing image pair for sample: {sample_name}")
    
    # Read images
    dapi_image = read_tiff(dapi_path)
    fitc_image = read_tiff(fitc_path)
    
    # Get metadata for normalization
    try:
        fitc_metadata = parse_metadata(extract_metadata(fitc_path))
    except:
        logger.warning(f"Could not extract metadata from {fitc_path}, using defaults")
        fitc_metadata = {
            "levelLow": 0,
            "levelHigh": 65535,
            "bitsPerSample": 16
        }
    
    # Normalize FITC image
    normalized_fitc = normalize_background(fitc_image)
    scaled_fitc = rescale_intensity(
        normalized_fitc,
        fitc_metadata.get("levelLow", 0),
        fitc_metadata.get("levelHigh", 65535),
        fitc_metadata.get("bitsPerSample", 16)
    )
    
    # Create cell segmentation from DAPI with enhanced segmentation methods
    cell_mask = create_mask(
        dapi_image, 
        thresholding_method="otsu", 
        min_area=min_cell_area,
        use_watershed=use_watershed,
        use_edge_detection=use_edge_detection,
        edge_sigma=1.0,
        watershed_footprint=5,
        watershed_compactness=0.1
    )
    
    # Label individual cells
    labeled_cells = measure.label(cell_mask)
    cell_regions = measure.regionprops(labeled_cells)
    
    # Process each cell
    cell_measurements = []
    all_foci_masks = np.zeros_like(cell_mask)
    
    for i, region in enumerate(cell_regions):
        # Create mask for this specific cell
        cell_id = i + 1
        single_cell_mask = labeled_cells == region.label
        
        # Detect foci in this cell
        foci_count, foci_mask = detect_foci(scaled_fitc, single_cell_mask, min_size=9)
        
        # Accumulate all foci masks
        all_foci_masks = np.logical_or(all_foci_masks, foci_mask)
        
        # Calculate total intensity normalized by area
        cell_area = region.area
        if cell_area > 0:
            total_intensity = np.sum(scaled_fitc[single_cell_mask])
            normalized_intensity = total_intensity / cell_area
        else:
            normalized_intensity = 0
        
        cell_measurements.append({
            'cell_id': cell_id,
            'foci_count': foci_count,
            'normalized_intensity': normalized_intensity,
            'cell_area': cell_area
        })
        
        logger.debug(f"Cell {cell_id}: {foci_count} foci, normalized intensity: {normalized_intensity:.2f}")
    
    # Save visualization if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization of cell segmentation and foci
        viz_path = os.path.join(output_dir, f"{sample_name}_visualization.png")
        save_visualization(dapi_image, scaled_fitc, cell_mask, all_foci_masks, viz_path)
        
        # Save segmentation steps visualization
        from pontifficate.segmentation import visualize_segmentation_steps
        steps_path = os.path.join(output_dir, f"{sample_name}_segmentation_steps.png")
        visualize_segmentation_steps(
            dapi_image,
            steps_path,
            thresholding_method="otsu",
            min_area=min_cell_area,
            use_watershed=use_watershed,
            use_edge_detection=use_edge_detection
        )
        
        # Save individual CSV for this sample
        csv_path = os.path.join(output_dir, f"{sample_name}_cell_analysis_results.csv")
        generate_sample_csv_report({
            'sample_name': sample_name,
            'cell_count': len(cell_regions),
            'cell_measurements': cell_measurements
        }, csv_path)
        
        logger.info(f"Analysis results for {sample_name} saved to {csv_path}")
    
    result = {
        'sample_name': sample_name,
        'cell_count': len(cell_regions),
        'cell_measurements': cell_measurements
    }
    
    return result


def process_directory(input_dir: str, output_dir: str, output_csv: str = None,
                     use_watershed: bool = True, use_edge_detection: bool = True,
                     min_cell_area: int = 100) -> List[Dict]:
    """
    Process all paired DAPI/FITC images in a directory.
    
    Args:
        input_dir (str): Directory containing paired images
        output_dir (str): Directory to save outputs
        output_csv (str, optional): Path to save combined CSV results
        use_watershed (bool): Whether to use watershed segmentation
        use_edge_detection (bool): Whether to use edge detection
        min_cell_area (int): Minimum cell area in pixels
        
    Returns:
        List[Dict]: Analysis results for all image pairs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image pairs
    pairs = find_image_pairs(input_dir)
    
    if not pairs:
        logger.warning(f"No DAPI/FITC image pairs found in {input_dir}")
        return []
    
    # Process each pair
    results = []
    
    for pair in pairs:
        try:
            result = analyze_image_pair(
                pair['dapi'], 
                pair['fitc'], 
                output_dir,
                use_watershed=use_watershed,
                use_edge_detection=use_edge_detection,
                min_cell_area=min_cell_area
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {pair['sample']}: {e}")
    
    # Generate combined CSV output if requested
    if output_csv:
        generate_combined_csv_report(results, output_csv)
        logger.info(f"Combined CSV report saved to {output_csv}")
    
    return results


def generate_sample_csv_report(result: Dict, output_path: str) -> None:
    """
    Generate a CSV report for a single sample.
    
    Args:
        result (Dict): Analysis result for a single sample
        output_path (str): Path to save CSV file
    """
    # Create a list to store all rows
    rows = []
    
    # Write header
    header = ['Sample', 'Cell ID', 'Foci Count', 'Normalized Intensity', 'Cell Area']
    rows.append(header)
    
    sample_name = result['sample_name']
    
    # If there are no cells, add a row with zeros
    if not result['cell_measurements']:
        rows.append([sample_name, 0, 0, 0, 0])
    else:
        # Add each cell's measurements as a separate row
        for cell in result['cell_measurements']:
            cell_id = cell['cell_id']
            foci_count = cell['foci_count']
            normalized_intensity = cell['normalized_intensity']
            cell_area = cell['cell_area']
            
            rows.append([sample_name, cell_id, foci_count, normalized_intensity, cell_area])
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    logger.debug(f"CSV report for {sample_name} saved to {output_path}")


def generate_combined_csv_report(results: List[Dict], output_path: str) -> None:
    """
    Generate a combined CSV report from analysis results for all samples.
    
    Args:
        results (List[Dict]): Analysis results for all samples
        output_path (str): Path to save CSV file
    """
    # Create a list to store all rows
    rows = []
    
    # Write header
    header = ['Sample', 'Cell ID', 'Foci Count', 'Normalized Intensity', 'Cell Area']
    rows.append(header)
    
    # Process each sample and add data for each individual cell
    for result in results:
        sample_name = result['sample_name']
        
        # If there are no cells, add a row with zeros
        if not result['cell_measurements']:
            rows.append([sample_name, 0, 0, 0, 0])
            continue
        
        # Add each cell's measurements as a separate row
        for cell in result['cell_measurements']:
            cell_id = cell['cell_id']
            foci_count = cell['foci_count']
            normalized_intensity = cell['normalized_intensity']
            cell_area = cell['cell_area']
            
            rows.append([sample_name, cell_id, foci_count, normalized_intensity, cell_area])
    
    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    logger.info(f"Combined CSV report saved to {output_path}")


def save_visualization(dapi: np.ndarray, fitc: np.ndarray, cell_mask: np.ndarray, 
                       foci_mask: np.ndarray, output_path: str) -> None:
    """
    Save a visualization of the segmentation and foci detection.
    
    Args:
        dapi (np.ndarray): DAPI image
        fitc (np.ndarray): FITC image
        cell_mask (np.ndarray): Cell segmentation mask
        foci_mask (np.ndarray): Detected foci mask
        output_path (str): Path to save the visualization
    """
    # Normalize images for display
    dapi_display = (dapi / dapi.max()) if dapi.max() > 0 else dapi
    fitc_display = (fitc / fitc.max()) if fitc.max() > 0 else fitc
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Display DAPI image
    axes[0, 0].imshow(dapi_display, cmap='gray')
    axes[0, 0].set_title('DAPI')
    axes[0, 0].axis('off')
    
    # Display FITC image
    axes[0, 1].imshow(fitc_display, cmap='gray')
    axes[0, 1].set_title('FITC')
    axes[0, 1].axis('off')
    
    # Display cell segmentation
    axes[1, 0].imshow(fitc_display, cmap='gray')
    cell_boundaries = np.zeros_like(cell_mask)
    
    # Create boundary mask and get cell properties for labeling
    labeled_cells = measure.label(cell_mask)
    regions = measure.regionprops(labeled_cells)
    
    # Create boundary mask
    for region in regions:
        coords = region.coords
        for coord in coords:
            y, x = coord
            # Check if the pixel is on the boundary
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < cell_mask.shape[0] and 
                        0 <= nx < cell_mask.shape[1] and 
                        labeled_cells[ny, nx] != region.label):
                        cell_boundaries[y, x] = 1
                        break
    
    # Overlay cell boundaries in red
    overlay = np.zeros((*cell_mask.shape, 4))
    overlay[..., 0] = 1.0  # Red channel
    overlay[..., 3] = cell_boundaries  # Alpha channel
    
    axes[1, 0].imshow(overlay, alpha=0.5)
    
    # Add cell ID labels to centroids
    for i, region in enumerate(regions):
        # Get centroid coordinates
        y, x = region.centroid
        cell_id = i + 1
        
        # Add text label with white text and black outline for visibility
        axes[1, 0].text(x, y, str(cell_id), 
                       color='white', fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round", 
                                 ec="black", 
                                 fc="black", 
                                 alpha=0.7))
    
    axes[1, 0].set_title('Cell Segmentation with Cell IDs')
    axes[1, 0].axis('off')
    
    # Display foci detection
    axes[1, 1].imshow(fitc_display, cmap='gray')
    
    # Overlay foci in green
    foci_overlay = np.zeros((*foci_mask.shape, 4))
    foci_overlay[..., 1] = 1.0  # Green channel
    foci_overlay[..., 3] = foci_mask  # Alpha channel
    
    axes[1, 1].imshow(foci_overlay, alpha=0.7)
    axes[1, 1].set_title('Detected Foci')
    axes[1, 1].axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Visualization saved to {output_path}") 