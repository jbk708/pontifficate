"""test_"""

import json

import numpy as np
import pytest
from tifffile import TiffFile, imwrite

from pontifficate.utils import read_tiff, save_tiff


def test_read_tiff(tmp_path):
    """Test reading a TIFF file."""
    file_path = tmp_path / "test_image.tif"
    mock_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    imwrite(file_path, mock_image)

    image = read_tiff(file_path)
    assert image.shape == mock_image.shape
    assert np.array_equal(image, mock_image)


def test_read_tiff_file_not_found():
    """Test reading a TIFF file that does not exist."""
    with pytest.raises(RuntimeError, match="Failed to read TIFF file"):
        read_tiff("nonexistent_file.tif")


def test_save_tiff_with_metadata(tmp_path):
    """Test saving a normalized image with metadata in a single TIFF tag."""
    file_path = tmp_path / "test_image_with_metadata.tiff"
    image = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    metadata = {"BitsPerSample": 16, "levelLow": 0.02, "levelHigh": 0.8}

    # Save the image
    save_tiff(image, file_path, metadata)

    with TiffFile(file_path) as tif:
        saved_image = tif.asarray()
        assert np.array_equal(saved_image, image)

        # Validate metadata
        tags = tif.pages[0].tags
        assert 65000 in tags  # Ensure the metadata tag exists
        saved_metadata = json.loads(tags[65000].value)
        assert saved_metadata == metadata
