#!/usr/bin/env python
import os

import pytest
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from nifti_processor import NIFTIProcessor, NIFTIOutputFile

from base_processor.tests import init_ssm
from base_processor.tests import setup_processor

test_dim_assignment_data = [
    # One dimensional nifti-1 file
    ('1d_nifti1.nii.gz', ['x']),
    # Two dimensional nifti-1 file
    ('2d_nifti1.nii.gz', ['x', 'y']),
    # Two dimensional nifti-1 file grayscale
    ('2d_grayscale_nifti1.nii.gz', ['x', 'y']),
    # Two dimensional nifti-1 file with a third dimension of dimensionality 3, implying RGB
    ('2d_color_nifti1.nii.gz', ['x', 'y', 'RGB']),
    # Three dimensional nifti-1 file, no dimension of dimensionality 3, implying grayscale
    ('3d_grayscale_nifti1.nii.gz', ['x', 'y', 'z']),
    # Three dimensional nifti-1 file: a simple grayscale volume despite
    # a second dimension of dimensionality 3 (meaningless)
    ('3d_non_color_nifti1.nii.gz', ['x', 'y', 'z']),
    # Four-dimensional nifti-1 file with a fourth dimension of dimensionality 3, implying RGB
    ('3d_color_nifti1.nii.gz', ['x', 'y', 'z', 'RGB']),
    # Five-dimensional nifti-1 file with a fourth dimension of dimensionality 3, implying RGB,
    # and last dimension automatically assumed to be time
    ('3d_color_time_nifti1.nii.gz', ['x', 'y', 'z', 'RGB', 'Time']),
    # Five-dimensional nifti-1 file with a fourth dimension not of dimensionality 3, implying grayscale,
    # and last dimension automatically assumed to be time
    ('3d_grayscale_time_nifti1.nii.gz', ['x', 'y', 'z', 'Time', 'Other']),
    # Four-dimensional nifti-1 file with no dimension of dimensionality 3
    ('4d_nifti1.nii.gz', ['x', 'y', 'z', 'Time']),
    # Five-dimensional nifti-1 file with no dimension of dimensionality 3
    ('5d_nifti1.nii.gz', ['x', 'y', 'z', 'Time', 'Other']),
    # Six-dimensional nifti-1 file with no dimension of dimensionality 3
    ('6d_nifti1.nii.gz', ['x', 'y', 'z', 'Time', 'Other', 'Other']),
    # Seven-dimensional nifti-1 file with no dimension of dimensionality 3
    ('7d_nifti1.nii.gz', ['x', 'y', 'z', 'Time', 'Other', 'Other'])
]


@pytest.mark.parametrize(
    "filename,expected,inputs",
    [
        (
                fn,
                expected,
                {
                    'file': os.path.join('/test-resources', fn)
                }
        ) for fn, expected in test_dim_assignment_data
    ],
    ids=[
        # One dimensional nifti-1 file
        '1d_nifti1',
        # Two dimensional nifti-1 file
        '2d_nifti1',
        # Two dimensional nifti-1 file grayscale
        '2d_grayscale_nifti1',
        # Two dimensional nifti-1 file with a third dimension of dimensionality 3, implying RGB
        '2d_color_nifti1',
        # Three dimensional nifti-1 file, no dimension of dimensionality 3, implying grayscale
        '3d_grayscale_nifti1',
        # Three dimensional nifti-1 file: a simple grayscale volume despite
        # a second dimension of dimensionality 3 (meaningless)
        '3d_non_color_nifti1',
        # Four-dimensional nifti-1 file with a fourth dimension of dimensionality 3, implying RGB
        '3d_color_nifti1',
        # Five-dimensional nifti-1 file with a fourth dimension of dimensionality 3, implying RGB,
        # and last dimension automatically assumed to be time
        '3d_color_time_nifti1',
        # Five-dimensional nifti-1 file with a fourth dimension not of dimensionality 3, implying grayscale,
        # and last dimension automatically assumed to be time
        '3d_grayscale_time_nifti1',
        # Four-dimensional nifti-1 file with no dimension of dimensionality 3
        '4d_nifti1',
        # Five-dimensional nifti-1 file with no dimension of dimensionality 3
        '5d_nifti1',
        # Six-dimensional nifti-1 file with no dimension of dimensionality 3
        '6d_nifti1',
        # Seven-dimensional nifti-1 file with no dimension of dimensionality 3
        '7d_nifti1'
    ]
)
def test_dim_assignment(filename, expected, inputs):
    print "~" * 60
    print " Using test file: %s to test dimension assignment" % filename
    print "~" * 60

    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    tsk = NIFTIProcessor(inputs=inputs)

    setup_processor(tsk)
    print "INPUTS = ", inputs

    # Load image
    output_file = NIFTIOutputFile()
    output_file.file_path = tsk.file
    output_file._load_image()

    assert output_file.get_dim_assignment() == expected

    # tsk._cleanup()

