"""segmentation.py"""

import cv2
import numpy as np
from skimage import filters, measure, morphology, feature, segmentation
from skimage.filters import gaussian
from scipy import ndimage as ndi

from pontifficate.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def create_mask(
    image: np.ndarray,
    thresholding_method: str = "otsu",
    min_area: int = 1,
    use_watershed: bool = False,
    use_edge_detection: bool = False,
    edge_sigma: float = 1.0,
    watershed_footprint: int = 3,
    watershed_compactness: float = 0.3,
) -> np.ndarray:
    """
    Create a binary mask for cell segmentation from an input image.

    Args:
        image (np.ndarray): Input grayscale image.
        thresholding_method (str): Method for thresholding ('otsu' or 'adaptive').
        min_area (int): Minimum area for objects to retain in the mask.
        use_watershed (bool): Whether to apply watershed segmentation to separate touching cells.
        use_edge_detection (bool): Whether to incorporate edge detection for improved boundaries.
        edge_sigma (float): Standard deviation for Gaussian filter used in edge detection.
        watershed_footprint (int): Size of footprint for watershed markers (larger values = fewer markers).
        watershed_compactness (float): Compactness parameter for watershed algorithm (higher = more regular shapes).

    Returns:
        np.ndarray: Binary mask with segmented cells.
    """
    logger.info("Starting mask creation...")

    # Normalize or rescale the image
    logger.info("Rescaling uint16 image to uint8...")
    rescaled_image = (image / image.max() * 255).astype(np.uint8)

    # Apply edge detection if requested
    if use_edge_detection:
        logger.info(f"Applying edge detection with sigma={edge_sigma}...")
        edges = feature.canny(rescaled_image, sigma=edge_sigma)
        # Convert to uint8 for visualization and further processing
        edge_image = (edges * 255).astype(np.uint8)

        # Use edges to enhance the original image
        # Subtract edges from the original to enhance boundaries
        enhanced_image = np.maximum(rescaled_image, edge_image)
    else:
        enhanced_image = rescaled_image

    # Apply thresholding
    if thresholding_method == "otsu":
        smoothed_image = gaussian(enhanced_image, sigma=2)
        logger.info("Applying Otsu's thresholding...")
        threshold = filters.threshold_otsu(smoothed_image)
        logger.info(f"Computed Otsu threshold: {threshold}")
        binary_mask = smoothed_image > threshold
    elif thresholding_method == "adaptive":
        logger.info("Applying adaptive thresholding...")
        binary_mask = (
            cv2.adaptiveThreshold(
                enhanced_image,
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

    # Apply watershed segmentation if requested
    if use_watershed:
        logger.info("Applying watershed segmentation to separate touching cells...")
        distance = ndi.distance_transform_edt(cleaned_mask)
        footprint = np.ones((watershed_footprint, watershed_footprint))

        # Modified peak_local_max call to be compatible with different scikit-image versions
        try:
            # Try the newer version first
            local_max = feature.peak_local_max(
                distance, indices=False, footprint=footprint, labels=cleaned_mask
            )
        except TypeError:
            # Fall back to older version if indices parameter isn't supported
            coords = feature.peak_local_max(
                distance, footprint=footprint, labels=cleaned_mask
            )
            # Create a boolean mask from coordinates
            local_max = np.zeros_like(distance, dtype=bool)
            local_max[tuple(coords.T)] = True
            logger.info("Using compatibility mode for peak_local_max")

        # Label the markers
        markers = ndi.label(local_max)[0]

        # Apply watershed
        logger.info(f"Running watershed with compactness={watershed_compactness}...")
        watershed_labels = segmentation.watershed(
            -distance, markers, mask=cleaned_mask, compactness=watershed_compactness
        )

        # Convert back to binary mask
        watershed_mask = watershed_labels > 0

        # Clean up the mask again
        final_mask = morphology.remove_small_objects(watershed_mask, min_size=min_area)
    else:
        final_mask = cleaned_mask

    logger.info("Mask creation complete.")
    return final_mask


def visualize_segmentation_steps(
    image: np.ndarray,
    output_path: str,
    thresholding_method: str = "otsu",
    min_area: int = 1,
    use_watershed: bool = False,
    use_edge_detection: bool = False,
) -> None:
    """
    Visualize the steps of the segmentation process and save to a file.

    Args:
        image (np.ndarray): Input grayscale image.
        output_path (str): Path to save the visualization.
        thresholding_method (str): Method for thresholding.
        min_area (int): Minimum area for objects.
        use_watershed (bool): Whether to use watershed segmentation.
        use_edge_detection (bool): Whether to use edge detection.
    """
    import matplotlib.pyplot as plt

    # Normalize for display
    display_img = (image / image.max()) if image.max() > 0 else image

    # Create the mask with only thresholding
    basic_mask = create_mask(
        image,
        thresholding_method=thresholding_method,
        min_area=min_area,
        use_watershed=False,
        use_edge_detection=False,
    )

    # Create mask with edge detection if requested
    if use_edge_detection:
        edge_mask = create_mask(
            image,
            thresholding_method=thresholding_method,
            min_area=min_area,
            use_watershed=False,
            use_edge_detection=True,
        )

    # Create mask with watershed if requested
    if use_watershed:
        watershed_mask = create_mask(
            image,
            thresholding_method=thresholding_method,
            min_area=min_area,
            use_watershed=True,
            use_edge_detection=use_edge_detection,
        )

    # Create figure with appropriate number of subplots
    n_plots = 2 + (1 if use_edge_detection else 0) + (1 if use_watershed else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5))

    # Display original image
    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Display basic mask
    axes[1].imshow(basic_mask, cmap="gray")
    axes[1].set_title("Basic Thresholding")
    axes[1].axis("off")

    # Display edge detection if used
    plot_idx = 2
    if use_edge_detection:
        axes[plot_idx].imshow(edge_mask, cmap="gray")
        axes[plot_idx].set_title("With Edge Detection")
        axes[plot_idx].axis("off")
        plot_idx += 1

    # Display watershed if used
    if use_watershed:
        axes[plot_idx].imshow(watershed_mask, cmap="gray")
        axes[plot_idx].set_title("With Watershed")
        axes[plot_idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Segmentation visualization saved to: {output_path}")
