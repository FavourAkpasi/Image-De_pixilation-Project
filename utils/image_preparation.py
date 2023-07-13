import glob
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")

    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]

    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def transform_center_crop(image):
    # compose, Apply transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((64, 64)),
    ])
    transformed_image = transform(image)

    return transformed_image


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    # Need a copy since we overwrite data directly
    image = image.copy()
    curr_x = x

    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size

    return image


def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 3 or image.shape[-3] != 1:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (..., 1, H, W)")
        pass
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")

    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (..., slice(y, y + height), slice(x, x + width))

    # This returns already a copy, so we are independent of "image"
    pixelated_image = pixelate(image, x, y, width, height, size)
    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False

    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image[area].copy()

    return pixelated_image, known_array, target_array


class ImagePreprocessing(Dataset):

    def __init__(
            self,
            image_dir: str,
            width_range: Tuple[int, int],
            height_range: Tuple[int, int],
            size_range: Tuple[int, int],
            dtype: Optional[type] = None,
    ):
        # The glob.glob function returns a list of file paths
        # that match the specified pattern
        self.image_files = sorted(
            [
                os.path.abspath(path)
                for path in glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)
            ]
        )
        # validate the images
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        if self.width_range[0] < 2:
            raise ValueError("Minimum width must be at least 2.")
        if self.width_range[0] > self.width_range[1]:
            raise ValueError("Minimum width cannot be greater than maximum width.")
        if self.height_range[0] < 2:
            raise ValueError("Minimum height must be at least 2.")
        if self.height_range[0] > self.height_range[1]:
            raise ValueError("Minimum height cannot be greater than maximum height.")
        if self.size_range[0] < 2:
            raise ValueError("Minimum size cannot be greater than maximum size.")
        if self.size_range[0] > self.size_range[1]:
            raise ValueError("Minimum size cannot be greater than maximum size.")

        self.dtype = dtype

    def __getitem__(self, index):
        rng = np.random.default_rng(seed=index)
        image_file = self.image_files[index]
        with Image.open(image_file) as img:
            cropped_image = transform_center_crop(img)
            image = np.array(cropped_image)
        if self.dtype is not None:
            image = image.astype(self.dtype)
        image = to_grayscale(image)
        # pixilated img and pixilated region size
        image_height, image_width = image.shape[1:]  # !channels
        width = rng.integers(*self.width_range)  # get val b/w min max
        height = rng.integers(*self.height_range)
        width = min(width, image_width)  #
        height = min(height, image_height)
        x = rng.integers(0, image_width - width + 1)
        y = rng.integers(0, image_height - height + 1)
        size = rng.integers(*self.size_range)
        pixelated_image, known_array, target_array = prepare_image(image, x, y, width, height, size)
        return pixelated_image, known_array, target_array, image_file

    def __len__(self):
        return len(self.image_files)

