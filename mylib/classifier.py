"""
Image classifier library
"""

import random
from pathlib import Path
from PIL import Image


def predict(image_path, class_names=None):
    """
    Predict the class of an image.

    Loads image from file or PIL Image object and returns predicted class.

    Parameters
    ----------
    image_path : str, Path, or PIL.Image
        Path to image file (str or Path object) or PIL Image object directly.
        Supported formats: JPG, PNG, BMP, GIF, TIFF.
    class_names : list of str, optional
        List of class names. Default: ['cardboard', 'paper', 'plastic', 'metal', 'trash', 'glass']

    Returns
    -------
    str
        Predicted class name (randomly selected from class_names).

    Raises
    ------
    FileNotFoundError
        If image file path does not exist.
    IOError
        If image file cannot be read.
    ValueError
        If image format is not supported or class_names is empty.

    Examples
    --------
    >>> predicted_class = predict("sample.jpg", ['cat', 'dog'])
    """
    if class_names is None:
        class_names = ["cardboard", "paper", "plastic", "metal", "trash", "glass"]

    if not class_names:
        raise ValueError("class_names cannot be empty")

    try:
        # Handle both file paths and PIL Images
        if isinstance(image_path, (str, Path)):
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image_path type: {type(image_path)}")

        # For Lab1: randomly select a class
        predicted_class = random.choice(class_names)
        return predicted_class

    except FileNotFoundError:
        raise
    except Exception as e:
        raise IOError(f"Error loading image: {str(e)}") from e


def resize(image_path, width, height):
    """
    Resize an image to specified dimensions.

    Parameters
    ----------
    image_path : str, Path, or PIL.Image
        Path to image file or PIL Image object.
    width : int
        Target width in pixels. Must be positive.
    height : int
        Target height in pixels. Must be positive.

    Returns
    -------
    tuple of (int, int)
        New dimensions (width, height) of the resized image.

    Raises
    ------
    FileNotFoundError
        If image file path does not exist.
    ValueError
        If width or height are not positive integers.
    IOError
        If image file cannot be read.

    Examples
    --------
    >>> new_size = resize("sample.jpg", 224, 224)
    >>> print(new_size)
    (224, 224)
    """
    if width <= 0:
        raise ValueError("'width' must be a positive integer")
    if height <= 0:
        raise ValueError("'height' must be a positive integer")

    try:
        # Handle both file paths and PIL Images
        if isinstance(image_path, (str, Path)):
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image_path type: {type(image_path)}")

        # Resize the image
        resized_image = image.resize((width, height))

        return resized_image.size  # Returns (width, height)

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        raise IOError(f"Error resizing image: {str(e)}") from e
