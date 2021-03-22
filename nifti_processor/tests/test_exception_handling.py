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

# Test file-based exceptions
test_file_exceptions_data = [
    ('1d_nifti1.nii.gz', ValueError)
]


@pytest.mark.parametrize(
    "filename,expected,inputs",
    [
        (
                fn,
                exp,
                {
                    'file': os.path.join('/test-resources', fn)
                }
        )
        for fn, exp in test_file_exceptions_data
    ],
    ids=[
        "ValueError_Degenerate_1d_Test"
    ]
)
def test_file_exception_handling(filename, expected, inputs):
    with pytest.raises(expected):
        print "~" * 60
        print " Using test file %s to check exception handling when dealing with invalid NIfTI-1 files " % filename
        print "~" * 60

        mock_ssm().start()
        mock_s3().start()

        init_ssm()

        # init task
        tsk = NIFTIProcessor(inputs=inputs)

        setup_processor(tsk)
        print "INPUTS = ", inputs

        # Run test
        tsk.run()

        # tsk._cleanup()


# Test internal exceptions
test_internal_exceptions_data = [
    ('2d_nifti1.nii.gz', np.uint8, ValueError, "Image has not been loaded yet into NIfTI processor."),
    ('2d_nifti1.nii.gz', 'blah', TypeError, "Image data type to convert to should be of type numpy.dtype")
]


@pytest.mark.parametrize(
    "filename, convert_format, expected, exception_text, inputs",
    [
        (
                fn,
                convert_fmt,
                exp,
                exp_text,
                {
                    'file': os.path.join('/test-resources', fn)
                }
        )
        for fn, convert_fmt, exp, exp_text in test_internal_exceptions_data
    ],
    ids=["No Image Loaded", "Wrong convert format"]
)
def test_internal_exception_handling(filename, convert_format, expected, exception_text, inputs):
    with pytest.raises(expected) as excinfo:
        print "~" * 60
        print " Using test file %s to check internal exception handling" % filename
        print "~" * 60

        mock_ssm().start()
        mock_s3().start()

        init_ssm()

        # init task
        tsk = NIFTIProcessor(inputs=inputs)
        # assert tsk.download_files is not None

        setup_processor(tsk)
        print "INPUTS = ", inputs

        # Run task
        if expected == ValueError:
            output_file = NIFTIOutputFile()
            output_file.file_path = tsk.file
            output_file._save_view()
        output_file = NIFTIOutputFile()
        output_file.file_path = tsk.file
        output_file._load_image()

        output_file._convert_image_data_type(format=convert_format)

    assert str(excinfo.value) == exception_text
