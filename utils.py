import numpy as np
import rawpy
from PIL import Image
from scipy.ndimage import convolve
from skimage import exposure


def is_raw_file(file_path):
    """Check if the file is a RAW image file supported by rawpy."""
    try:
        with rawpy.imread(file_path) as _:
            return True
    except:
        return False


def is_tiff_file(file_path):
    """Check if the file is a TIFF image file."""
    try:
        with Image.open(file_path) as img:
            return img.format == "TIFF" and file_path.lower().endswith(".tiff")
    except:
        return False


def invert_image(image):
    """Invert a 16-bit image."""
    return np.iinfo(image.dtype).max - image


def get_central_region(image, border_width):
    h, w = image.shape[:2]
    border_width = min(border_width, h // 2, w // 2)
    return image[border_width : h - border_width, border_width : w - border_width]


def crop_image(image, border_width_percentage):
    """Crop the image by a given percentage of its border width."""
    return get_central_region(
        image, int(min(image.shape[:2]) * border_width_percentage / 100)
    )


def noise_reduction(image, filter_size=2):
    """Perform noise reduction using a simple mean filter."""
    kernel = np.ones((filter_size, filter_size)) / (filter_size**2)
    smoothed_image = np.zeros_like(image)

    for i in range(image.shape[2]):  # Apply filter to each channel
        smoothed_image[..., i] = convolve(image[..., i], kernel, mode="reflect")

    return smoothed_image


def contrast_stretch(image, border_width_percentage):
    """Apply contrast stretching to the image based on the central region."""
    border_width = int(min(image.shape[:2]) * border_width_percentage / 100)
    central_region = get_central_region(image, border_width)

    lower_bound, upper_bound = np.percentile(central_region, [5, 99])
    if lower_bound == upper_bound:
        raise ValueError(
            "The central region has no contrast. Please adjust the border width."
        )

    return exposure.rescale_intensity(
        image, in_range=(lower_bound, upper_bound), out_range=(10, 65535)
    ).astype(np.uint16)


def white_balance(img, central_region):
    """Apply white balance to a 16-bit image using simple scaling."""
    avgR, avgG, avgB = np.mean(central_region, axis=(0, 1))
    kR = avgG / avgR
    kB = avgG / avgB
    img[:, :, 0] = np.clip(img[:, :, 0] * kR, 0, 65535)
    img[:, :, 2] = np.clip(img[:, :, 2] * kB, 0, 65535)
    return img
