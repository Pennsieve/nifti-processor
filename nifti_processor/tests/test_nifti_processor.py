#!/usr/bin/env python
import os
import json

import numpy as np
import pytest
from moto import mock_s3
from moto import mock_ssm
# our module(s)
from nifti_processor import NIFTIProcessor, NIFTIOutputFile

from base_processor.tests import init_ssm, setup_processor

test_processor_data = [
    '2d_nifti1.nii.gz',
    # '2d_color_nifti1.nii.gz',
    # '3d_grayscale_nifti1.nii.gz',
    # '3d_color_nifti1.nii.gz',
    # '3d_color_time_nifti1.nii.gz',
    # '3d_grayscale_time_nifti1.nii.gz'
]


@pytest.mark.parametrize("filename", test_processor_data)
def test_explode_assets(filename):
    mock_ssm().start()
    mock_s3().start()

    init_ssm()

    # init task
    inputs = {'file': os.path.join('/test-resources', filename)}
    task = NIFTIProcessor(inputs=inputs)

    setup_processor(task)

    # run
    task.run()

    print task.outputs[0].file_size
    print task.outputs[0].view_size

    # Ensure task completed
    assert os.path.isfile('view_asset_info.json')
    json_dict = json.load(open('view_asset_info.json'))
    assert 'size' in json_dict.keys()
    assert 'fileType' in json_dict.keys()

    assert os.path.exists('view')
    assert os.path.exists(
        os.path.join(
            'view',
            'dimensions0.json'
        )
    )

    # Ensure task generated a numpy array data matrix
    assert isinstance(task.outputs[0].img_data, np.ndarray)

    # Ensure correct data type extracted from data matrix
    assert task.outputs[0].img_data.dtype == task.outputs[0].img_data_dtype

    # Clean up
    os.system('rm view/*.png')
    os.system('rm view/*.dzi')
    os.system('rm view/*_files')
