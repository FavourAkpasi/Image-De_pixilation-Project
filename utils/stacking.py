# -*- coding: utf-8 -*-
"""
Author -- Favour Akpasi
stacking functions of Depixilation project.
"""

import numpy as np
from typing import List, Tuple


def stack_with_padding(batch_as_list: List[Tuple]) -> Tuple:
    batch_size = int(len(batch_as_list))
    # Get maximum width and height of all images in the batch
    max_height = max(image.shape[1] for image, _, _, _ in batch_as_list)
    max_width = max(image.shape[2] for image, _, _, _ in batch_as_list)

    # Stack pixelated images
    stacked_pixelated_images = np.zeros((batch_size, 1, max_height, max_width))
    stacked_known_arrays = np.zeros((batch_size, 1, max_height, max_width))
    stacked_target_arrays = np.zeros((batch_size, 1, max_height, max_width))
    image_files = []

    for i, (img, kno, tar, image_file) in enumerate(batch_as_list):
        stacked_pixelated_images[i, :, :img.shape[1], :img.shape[2]] = img
        stacked_known_arrays[i, :, :kno.shape[1], :kno.shape[2]] = kno
        stacked_target_arrays[i, :, :tar.shape[1], :tar.shape[2]] = tar
        image_files.append(image_file)

    # Concatenate channels of pixelated_images and known_arrays
    stacked_pixelated_images = np.concatenate(stacked_pixelated_images, axis=1)
    stacked_known_arrays = np.concatenate(stacked_known_arrays, axis=1)

    return stacked_pixelated_images, stacked_known_arrays, stacked_target_arrays, image_files
