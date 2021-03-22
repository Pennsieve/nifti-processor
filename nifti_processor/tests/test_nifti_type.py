#!/usr/bin/env python
import os

import nibabel as nib
import pytest
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from nifti_processor import NIFTIProcessor, NIFTIOutputFile

from base_processor.tests import init_ssm
from base_processor.tests import setup_processor

test_nifti_type_data = [
    ('3d_grayscale_nifti1.nii.gz', nib.nifti1.Nifti1Image),
    ('3d_nifti2.nii.gz', nib.nifti2.Nifti2Image)
]


@pytest.mark.parametrize(
    "filename,expected,inputs",
    [
        (
                fn,
                nifti_type,
                {
                    'file': os.path.join('/test-resources', fn)
                }
        )
        for fn, nifti_type in test_nifti_type_data
    ],
    ids=[
        '3d_nifti1',
        '3d_nifti2'
    ]
)
def test_nifti_type(filename, expected, inputs):
    print "~" * 60
    print " Using test file %s to check type of ingest file as NIftI-1 or NIfTI-2 " % filename
    print "~" * 60

    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    tsk = NIFTIProcessor(inputs=inputs)

    setup_processor(tsk)
    print "INPUTS = ", inputs

    output_file = NIFTIOutputFile()
    output_file.file_path = tsk.file
    output_file._load_image()

    assert type(output_file.img_nii) == expected

    # tsk._cleanup()
