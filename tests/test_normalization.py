"""test_normalization.py"""

import numpy as np
import pytest

from pontifficate.normalization import (
    normalize_background,
    rescale_intensity,
)


def test_rescale_intensity_16bit():
    """Test intensity rescaling for 16-bit images."""
    image = np.array([[1000, 2000, 3000], [4000, 5000, 65535]], dtype=np.uint16)
    level_low = 0.1
    level_high = 0.9

    rescaled = rescale_intensity(image, level_low, level_high, bits_per_sample=16)
    assert np.min(rescaled) == 0
    assert np.max(rescaled) == 65535


def test_rescale_intensity_missing_levels():
    """Test intensity rescaling when levelLow or levelHigh is missing."""
    image = np.array([[50, 100, 150], [200, 250, 255]], dtype=np.uint16)

    # Missing levelLow and levelHigh
    rescaled = rescale_intensity(image, None, None)
    assert np.array_equal(image, rescaled)  # Should return the original image


def test_rescale_intensity_different_bits_per_sample():
    """Test intensity rescaling with different bits per sample."""
    image = np.array([[1000, 2000, 3000], [4000, 5000, 65535]], dtype=np.uint16)
    level_low = 0.1
    level_high = 0.9
    bits_per_sample = 16

    rescaled = rescale_intensity(image, level_low, level_high, bits_per_sample)
    assert np.min(rescaled) == 0
    assert np.max(rescaled) == 65535


def test_normalize_background_valid():
    """Test background normalization with valid inputs."""
    image = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    normalized = normalize_background(image)
    assert np.min(normalized) == 0
    assert np.max(normalized) == 65535
