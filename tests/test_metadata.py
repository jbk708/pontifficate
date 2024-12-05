"""test_metadata.py"""

import numpy as np
import pytest
from tifffile import imwrite

from pontifficate.metadata import extract_metadata, parse_metadata


def test_extract_metadata_valid():
    """Test extracting valid metadata from ExifTag (34665)."""
    file_path = "data/Image_0001_tritc.tiff"

    metadata = extract_metadata(file_path)
    assert isinstance(metadata, dict)
    assert metadata["DateTimeOriginal"] == "2024:11:14 16:57:30"
    assert "UserComment" in metadata
    assert metadata["bitsPerSample"] == 16


def test_extract_metadata_missing(tmp_path):
    """Test extracting metadata when ExifTag (34665) is missing."""
    file_path = tmp_path / "test_missing_metadata_image.tif"
    mock_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    imwrite(file_path, mock_image)

    with pytest.raises(RuntimeError, match="Failed to extract metadata from TIFF file"):
        extract_metadata(file_path)


def test_extract_metadata_invalid(tmp_path):
    """Test extracting metadata when ExifTag (34665) contains invalid data."""
    file_path = tmp_path / "test_invalid_metadata_image.tif"
    mock_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    invalid_metadata = "Not a valid dictionary"

    extra_tags = [(34665, "s", 1, invalid_metadata, True)]
    imwrite(file_path, mock_image, extratags=extra_tags)

    with pytest.raises(RuntimeError, match="Failed to extract metadata from TIFF file"):
        extract_metadata(file_path)


def test_parse_metadata_valid_flattened():
    """Test parsing valid metadata with flattened imageCreationSummary."""
    metadata = {
        "bitsPerSample": 16,
        "UserComment": '{"microscopeMode": 1, "imageCode": 2, "effectivePixelSize": 3.5509, "objectiveMag": 40, "imageCreationSummary": {"ledIntensity": 67, "levelLow": 0.02, "gain": 8, "levelHigh": 0.37, "physicalChannelName": "TRITC", "exposure": 100, "dhr": {"intensity": 0, "version": 1}, "levelMid": 1}}',
    }
    parsed = parse_metadata(metadata)
    assert parsed["bitsPerSample"] == 16
    assert parsed["microscopeMode"] == 1
    assert parsed["imageCode"] == 2
    assert parsed["effectivePixelSize"] == 3.5509
    assert parsed["objectiveMag"] == 40
    assert parsed["ledIntensity"] == 67
    assert parsed["levelLow"] == 0.02
    assert parsed["gain"] == 8
    assert parsed["levelHigh"] == 0.37
    assert parsed["physicalChannelName"] == "TRITC"
    assert parsed["exposure"] == 100
    assert parsed["dhrIntensity"] == 0
    assert parsed["dhrVersion"] == 1
    assert parsed["levelMid"] == 1


def test_parse_metadata_missing_image_creation_summary():
    """Test parsing metadata with missing imageCreationSummary."""
    metadata = {
        "bitsPerSample": 16,
        "UserComment": '{"microscopeMode": 1, "imageCode": 2, "effectivePixelSize": 3.5509, "objectiveMag": 40}',
    }
    parsed = parse_metadata(metadata)
    assert parsed["bitsPerSample"] == 16
    assert parsed["microscopeMode"] == 1
    assert parsed["imageCode"] == 2
    assert parsed["effectivePixelSize"] == 3.5509
    assert parsed["objectiveMag"] == 40
    assert parsed["ledIntensity"] is None
    assert parsed["levelLow"] is None
    assert parsed["gain"] is None
    assert parsed["levelHigh"] is None
    assert parsed["physicalChannelName"] is None
    assert parsed["exposure"] is None
    assert parsed["dhrIntensity"] is None
    assert parsed["dhrVersion"] is None
    assert parsed["levelMid"] is None


def test_parse_metadata_missing_usercomment():
    """Test parsing metadata when UserComment is missing."""
    metadata = {}
    with pytest.raises(ValueError):
        parse_metadata(metadata)


def test_parse_metadata_invalid_usercomment():
    """Test parsing metadata with invalid JSON in UserComment."""
    metadata = {"UserComment": "invalid JSON"}
    with pytest.raises(ValueError):
        parse_metadata(metadata)


def test_parse_metadata_unexpected_error():
    """Test parsing metadata with unexpected error."""
    metadata = "Not a dictionary"  # Invalid type
    with pytest.raises(ValueError, match="Failed to parse metadata."):
        parse_metadata(metadata)
