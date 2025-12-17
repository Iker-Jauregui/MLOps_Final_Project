"""
Unit Testing of the application's logic
"""
import pytest
from pathlib import Path
from PIL import Image
from mylib.classifier import predict, resize


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary sample image for testing."""
    img = Image.new('RGB', (100, 100), color='blue')
    img_path = tmp_path / "sample.jpg"
    img.save(img_path)
    return str(img_path)


def test_predict_with_file_path(sample_image_path):
    """Test predict function with file path."""
    result = predict(sample_image_path, ['cat', 'dog'])
    assert result in ['cat', 'dog']


def test_predict_with_pil_image():
    """Test predict function with PIL Image."""
    img = Image.new('RGB', (100, 100), color='green')
    result = predict(img, ['cat', 'dog', 'bird'])
    assert result in ['cat', 'dog', 'bird']


def test_predict_default_classes(sample_image_path):
    """Test predict with default class names."""
    result = predict(sample_image_path)
    default_classes = ['cardboard', 'paper', 'plastic', 'metal', 'trash', 'glass']
    assert result in default_classes


def test_predict_file_not_found():
    """Test predict with non-existent file."""
    with pytest.raises(FileNotFoundError):
        predict("nonexistent.jpg", ['cat'])


def test_resize_with_file_path(sample_image_path):
    """Test resize function with file path."""
    result = resize(sample_image_path, 32, 32)
    assert result == (32, 32)


def test_resize_with_pil_image():
    """Test resize function with PIL Image."""
    img = Image.new('RGB', (100, 100), color='yellow')
    result = resize(img, 64, 64)
    assert result == (64, 64)


def test_resize_invalid_width(sample_image_path):
    """Test resize with invalid width."""
    with pytest.raises(ValueError, match="'width' must be a positive integer"):
        resize(sample_image_path, 0, 32)


def test_resize_invalid_height(sample_image_path):
    """Test resize with invalid height."""
    with pytest.raises(ValueError, match="'height' must be a positive integer"):
        resize(sample_image_path, 32, -5)


def test_resize_file_not_found():
    """Test resize with non-existent file."""
    with pytest.raises(FileNotFoundError):
        resize("nonexistent.jpg", 32, 32)