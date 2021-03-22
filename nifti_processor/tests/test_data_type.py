#!/usr/bin/env python
import os

import numpy as np
import pytest
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from nifti_processor import NIFTIProcessor, NIFTIOutputFile

from base_processor.tests import init_ssm
from base_processor.tests import setup_processor

test_data_type_data = [
    (
        fn,
        np.dtype(fn.split('_')[1]),
        {
            'file': os.path.join('/test-resources', fn)
        }
    )
    for fn in ['type_complex128_nifti1.nii.gz', 'type_complex64_nifti1.nii.gz', 'type_float32_nifti1.nii.gz',
               'type_float64_nifti1.nii.gz', 'type_int16_nifti1.nii.gz', 'type_int32_nifti1.nii.gz',
               'type_int64_nifti1.nii.gz', 'type_int8_nifti1.nii.gz', 'type_uint16_nifti1.nii.gz',
               'type_uint32_nifti1.nii.gz', 'type_uint64_nifti1.nii.gz', 'type_uint8_nifti1.nii.gz']
]


@pytest.mark.parametrize(
    "filename,expected,inputs",
    test_data_type_data,
    ids=[
        'complex128_nifti1',
        'complex64_nifti1',
        'float32_nifti1',
        'float64_nifti1',
        'int16_nifti1',
        'int32_nifti1',
        'int64_nifti1',
        'int8_nifti1',
        'uint16_nifti1',
        'uint32_nifti1',
        'uint64_nifti1',
        'uint8_nifti1'
    ]
)
def test_data_type(filename, expected, inputs):
    print "~" * 60
    print " Using test file %s to confirm has data type %s" % (filename, expected)
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

    assert output_file.img_data_dtype == expected

    # tsk._cleanup()


test_convert_image_type_data = [
    (
        fn,
        expected_type,
        {
            'file': os.path.join('/test-resources', fn)
        }
    )
    for fn, expected_type in [
        ('type_uint8_nifti1.nii.gz', np.uint8),
        ('type_uint16_nifti1.nii.gz', np.uint8)
    ]
]


@pytest.mark.parametrize(
    "filename,expected,inputs",
    test_convert_image_type_data,
    ids=[
        'uint8_no_need_for_conversion',
        'uint16_need_to_convert'
    ]
)
def test_convert_image_data_type(filename, expected, inputs):
    print "~" * 60
    print " Using test file %s to test _convert_image_data_type() and check conversion to uint8" % filename
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

    # Convert type of image in order to save in deep-zoom
    output_file._convert_image_data_type()

    assert output_file.img_data_dtype == expected

    # tsk._cleanup()
