#!/usr/bin/env python
import glob
import itertools
import os

import numpy as np
import pytest
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from nifti_processor import NIFTIProcessor, NIFTIOutputFile

from base_processor.tests import init_ssm

test_processor_data = [
    '2d_nifti1.nii.gz',
    '2d_color_nifti1.nii.gz',
    '3d_grayscale_nifti1.nii.gz',
    '3d_color_nifti1.nii.gz',
    '3d_color_time_nifti1.nii.gz',
    '3d_grayscale_time_nifti1.nii.gz'
]


@pytest.mark.parametrize("filename", test_processor_data)
def test_explode_assets(filename):
    inputs = {'file': os.path.join('/test-resources', filename)}

    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    task = NIFTIProcessor(inputs=inputs)
    output_file = NIFTIOutputFile()
    output_file.file_path = task.file

    # Load image
    output_file._load_image()

    # Convert type of image in order to save in deep-zoom
    if output_file.img_data_dtype != np.uint8:
        output_file._convert_image_data_type()

        output_file._save_view(exploded_format='png')

    # Get number of files generated
    num_png_files = len(glob.glob('view/*.png'))

    # Ensure correct number of png outputs
    if output_file.hasRGBDimension:
        # Compute combinatorial number of files expected
        shape = list(output_file.img_data.shape)
        # Remove RGB dimension from combinatorial count
        shape.pop(output_file.RGBDimension)
    else:
        # Compute combinatorial number of files expected
        shape = list(output_file.img_data.shape)

    total_count = 0
    for j in itertools.combinations(shape, len(shape) - 2):
        total_count += np.prod(j)

    # Make sure same number of files generated
    assert num_png_files == total_count

    # Save to local storage output deepzoom files
    output_file._save_view(exploded_format='dzi')

    # Get number of files generated
    num_dzi_files = len(glob.glob('view/*.dzi'))

    # Ensure correct number of dzi outputs
    if output_file.hasRGBDimension:
        # Compute combinatorial number of files expected
        shape = list(output_file.img_data.shape)
        # Remove RGB dimension from combinatorial count
        shape.pop(output_file.RGBDimension)
    else:
        # Compute combinatorial number of files expected
        shape = list(output_file.img_data.shape)

    total_count = 0
    for j in itertools.combinations(shape, len(shape) - 2):
        total_count += np.prod(j)

    # Make sure same number of files generated
    assert num_dzi_files == total_count

    # Clean up
    os.system('rm -rf view')
