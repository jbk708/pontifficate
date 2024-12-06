"""segmentation.py"""

import numpy as np

from pontifficate.segmentation import create_mask, draw_boundaries, segment_cells


def test_create_mask_otsu():
    """Test mask creation using Otsu's thresholding."""
    # Create a synthetic image with distinct intensity regions
    image = np.array(
        [
            [0, 0, 0, 255, 255],
            [0, 0, 0, 255, 255],
            [0, 0, 0, 255, 255],
            [255, 255, 255, 0, 0],
            [255, 255, 255, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask = create_mask(image, thresholding_method="otsu", min_area=5)
    assert mask.sum() > 0
    assert mask.shape == image.shape


def test_create_mask_adaptive():
    """Test mask creation using adaptive thresholding."""
    image = np.linspace(0, 255, 25).reshape(5, 5).astype(np.uint8)

    mask = create_mask(image, thresholding_method="adaptive", min_area=1)
    assert mask.sum() > 0
    assert mask.shape == image.shape


def test_segment_cells():
    """Test cell segmentation from a binary mask."""
    mask = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
    regions = segment_cells(mask)

    assert len(regions) == 2  # Expect two distinct regions
    assert all(
        region.area > 0 for region in regions
    )  # Regions should have non-zero area


def test_draw_boundaries():
    """Test boundary overlay on an image."""
    image = np.zeros((5, 5), dtype=np.uint8)
    mask = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=bool,
    )

    boundary_image = draw_boundaries(image, mask)
    assert boundary_image.shape == image.shape  # Ensure dimensions match
    assert np.any(boundary_image > image)  # Boundaries should modify the image
