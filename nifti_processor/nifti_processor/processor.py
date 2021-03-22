import itertools
import json
import os

import PIL.Image
import boto3
import nibabel as nib
import numpy as np
from base_processor.imaging import utils
from botocore.client import Config

from base_image_radiology_processor import BaseRadiologyImageProcessor


class NIFTIOutputFile(object):
    def __init__(self, optimize=False, *args, **kwargs):
        self.file_path = None
        self.view_path = None
        self.img_dimensions = {}
        self.num_dimensions = -1

        self.img_nii = None
        self.img_hdr = None
        self.img_affine = None
        self.img_data = None
        self.img_data_shape = None
        self.img_data_dtype = None

        self.primary_view_dimensions = (-1, -1)
        self.hasRGBDimension = False
        self.RGBDimension = -1
        self.hasTimeDimension = False
        self.TimeDimension = -1

        self.view_format = 'png'
        self.optimize = kwargs.get('optimize', False)
        self.tile_size = kwargs.get('tile_size', 128)
        self.tile_overlap = kwargs.get('tile_overlap', 0)
        self.tile_format = kwargs.get('tile_format', "png")
        self.image_quality = kwargs.get('image_quality', 1.0)
        self.resize_filter = kwargs.get('resize_filter', "bicubic")

    @property
    def file_size(self):
        return os.path.getsize(self.file_path)

    @property
    def view_size(self):
        return os.path.getsize(self.view_path)

    def get_view_asset_dict(self, storage_bucket, upload_key):
        upload_key = upload_key.rstrip('/')
        json_dict = {
            "bucket": storage_bucket,
            "key": upload_key,
            "type": "View",
            "size": self.file_size,
            "fileType": "NIFTI"
        }
        return json_dict

    def get_dim_assignment(self):
        """Retrieve inferred dimension assignment based on number of dimensions and length of each dimension. """

        # TODO: Include "intelligence" to determine RGBA channels (length 4) in addition to RGB (length 3)
        # Check and set dimension if contains RGB channel
        possible_color_channels = np.where(np.array(self.img_data.shape) == 3)[0]

        if self.num_dimensions == 1:
            # Degenerate case
            return ['x']
        elif self.num_dimensions == 2:
            return ['x', 'y']
        elif self.num_dimensions == 3:
            if np.count_nonzero(possible_color_channels) != 1:
                # Simple 3D image unlikely to have more than 2 RGB channels
                return ['x', 'y', 'z']
            elif np.count_nonzero(possible_color_channels) == 1:
                if possible_color_channels[0] == 3 - 1:
                    return ['x', 'y', 'RGB']
                else:  # Unlikely to be color
                    return ['x', 'y', 'z']
        else:
            # Large dimension matrices are likely to follow convention: x, y, z, color, time, Other, Other, ...
            default_dim_assignment = ['x', 'y', 'z'] + ['Other'] * (self.num_dimensions - 3)
            # Check if expected color channel is truly RGB, else set to Time
            if self.img_data.shape[3] == 3:
                default_dim_assignment[3] = 'RGB'
                try:
                    default_dim_assignment[4] = 'Time'
                except IndexError:
                    pass
            else:
                default_dim_assignment[3] = 'Time'
            return default_dim_assignment

    def set_img_dimensions(self):
        dim_assignment = self.get_dim_assignment()

        self.img_dimensions['num_dimensions'] = self.num_dimensions
        self.img_dimensions['isColorImage'] = False
        self.img_dimensions['dimensions'] = {}

        for dim in range(self.num_dimensions):
            self.img_dimensions['dimensions'][dim] = {}
            self.img_dimensions['dimensions'][dim]["name"] = 'dimension_%s' % str(dim)
            if dim_assignment[dim].lower() == 'x':
                self.img_dimensions['dimensions'][dim]["assignment"] = 'spatial_x'
            elif dim_assignment[dim].lower() == 'y':
                self.img_dimensions['dimensions'][dim]["assignment"] = 'spatial_y'
            elif dim_assignment[dim].lower() == 'z':
                self.img_dimensions['dimensions'][dim]["assignment"] = 'spatial_z'
            elif dim_assignment[dim].lower() == 'c' or dim_assignment[dim].lower() == 'rgb':
                self.img_dimensions['dimensions'][dim]["assignment"] = 'color'
            elif dim_assignment[dim].lower() == 't':
                self.img_dimensions['dimensions'][dim]["assignment"] = 'time'
            else:
                self.img_dimensions['dimensions'][dim]["assignment"] = 'other'
            self.img_dimensions['dimensions'][dim]["length"] = self.img_data.shape[dim]
            self.img_dimensions['dimensions'][dim]["resolution"] = float(self.img_hdr.get_zooms()[dim])
            self.img_dimensions['dimensions'][dim]["unit"] = "mm"
            if dim_assignment[dim] == 'RGB':
                self.hasRGBDimension = True
                self.RGBDimension = dim
            if dim_assignment[dim] == 'Time':
                self.hasTimeDimension = True
                self.TimeDimension = dim
        self.img_dimensions['isColorImage'] = self.hasRGBDimension

    def _load_image(self):
        # Load image
        # try:
        self.img_nii = nib.load(self.file_path)
        # except ImageFileError:
        #     raise

        # Get header
        self.img_hdr = self.img_nii.header
        # if self.img_hdr is None:
        #     raise ValueError('Image header is not set properly')

        # Get image affine matrix
        self.img_affine = self.img_nii.affine
        # if self.img_affine is None:
        #     raise ValueError('Image affine matrix is empty')

        # Get image data matrix
        self.img_data = self.img_nii.get_data().squeeze()
        self.img_data_shape = self.img_data.shape

        # if self.img_data is None or self.img_data == np.array(()):
        #     raise ValueError('Image data matrix is empty')

        # Get image data matrix data type
        self.img_data_dtype = self.img_data.dtype
        # if self.img_data_dtype is None:
        #     raise ValueError('Image data matrix data type is not assigned')

        # Set number of dimensions of image matrix
        self.num_dimensions = len(self.img_data.shape)

        # Set image dimension/"channel" information
        # output file is just the input file in NIfTI case
        self.set_img_dimensions()

    def load_image(self, nifti_file_path):
        # Set file path
        self.file_path = nifti_file_path

        # Load NIfTI Image
        self._load_image()

        # Convert type of image in order to save in deep-zoom
        if self.img_data_dtype != np.uint8:
            self._convert_image_data_type()
            assert self.img_data_dtype == np.uint8

        # Save to local storage output deepzoom files
        if np.prod(self.img_data_shape) > 1E8:
            self._save_view(exploded_format='dzi')
            self.view_format = 'dzi'
        else:
            self._save_view(exploded_format='png')
            self.view_format = 'png'

    def _convert_image_data_type(self, format=np.uint8):
        """ Convert image data matrix datatype to another format"""

        # Check correct data type format to convert
        try:
            if type(format) == str:
                format = np.dtype(format)
        except TypeError:
            raise TypeError("Image data type to convert to should be of type numpy.dtype")

        # Convert only if image data type is not in desired data type format
        if self.img_data_dtype != format:
            self.img_data = utils.convert_image_data_type(self.img_data, format)
            self.img_data_dtype = format
        return

    def _save_view(self, exploded_format='dzi'):
        """ Save exploded assets from the image in internal storage format to output local path.
        These files will be uploaded to S3"""

        # Make view directory
        if not os.path.exists('view'):
            os.makedirs('view')

        # Check _load_image() has been run to load NIfTI file
        if self.num_dimensions == -1:
            raise ValueError("Image has not been loaded yet into NIfTI processor.")

        # Degenerate NIfTI file
        elif self.num_dimensions == 1:
            raise ValueError("Image is degenerate: has only one dimension.")

        # Simple two-dimensional NIfTI file
        elif self.num_dimensions == 2:
            # Generate image object
            image = PIL.Image.fromarray(np.swapaxes(self.img_data, 0, 1))

            # Save asset in appropriate format
            filename = os.path.join(
                'view',
                # Assumed that ingested file will have .nii.gz or .nii as extensions
                os.path.basename(self.file_path).replace('.nii.gz', '.%s' % exploded_format).replace(
                    '.nii', '.%s' % exploded_format)
            )
            utils.save_asset(
                image,
                exploded_format,
                filename,
                optimize=self.optimize, tile_size=self.tile_size,
                tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                image_quality=self.image_quality, resize_filter=self.resize_filter
            )

        # Three-dimensional NIfTI file can be either:
        #   1. 3-D Volume
        #   2. 2-D RGB Plane image
        elif self.num_dimensions == 3:
            # Check if RGB, in which case, save exploded assets as RGB images
            if self.hasRGBDimension:
                # TODO: Make sure last dimension is RGB channel; re-arrange if not

                # Generate image object
                image = PIL.Image.fromarray(np.swapaxes(self.img_data, 0, 1), "RGB")

                # Save asset in appropriate format
                filename = os.path.join(
                    'view',
                    # Assumed that ingested file will have .nii.gz or .nii as extensions
                    os.path.basename(self.file_path).replace('.nii.gz', '.%s' % exploded_format).replace(
                        '.nii', '.%s' % exploded_format)
                )
                utils.save_asset(
                    image,
                    exploded_format,
                    filename,
                    optimize=self.optimize, tile_size=self.tile_size,
                    tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                    image_quality=self.image_quality, resize_filter=self.resize_filter
                )
            else:
                # Explode deep-zoom assets by slicing in every dimension
                for dim in range(self.num_dimensions):
                    # Iterate over each singleslice along a given dimension dim
                    for ii in range(self.img_dimensions['dimensions'][dim]['length']):
                        # Generate image object
                        sliced_image = utils.singleslice(self.img_data, ii, dim)
                        image = PIL.Image.fromarray(np.flip(np.swapaxes(sliced_image, 0, 1), 0))

                        # Save asset in appropriate format
                        filename = os.path.join(
                            'view',
                            'dim_{dim}_slice_{slice}.{fmt}'.format(dim=dim, slice=ii, fmt=exploded_format)
                        )

                        utils.save_asset(
                            image,
                            exploded_format,
                            filename,
                            optimize=self.optimize, tile_size=self.tile_size,
                            tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                            image_quality=self.image_quality, resize_filter=self.resize_filter
                        )

        # High-dimensional NIfTI file can be either:
        #   1. N-dimensional HyperVolume image
        #   2. (N-1)-dimensional RGB HyperVolume image
        elif self.num_dimensions >= 4:
            # Check if RGB, in which case, save exploded assets as RGB assets
            if self.hasRGBDimension:
                # Remove RGB dimension from iterator
                img_data_dimx = range(len(self.img_data.shape))
                img_data_dimx.pop(self.RGBDimension)

                # Explode deep-zoom assets by slicing in all but 3 dimensions (2D + RGB)
                for slice_dimx in itertools.combinations(img_data_dimx, self.num_dimensions - 3):
                    for slicex in utils.multiradix_recursive(np.array(self.img_data.shape)[list(slice_dimx)]):
                        # Generate image object
                        image = utils.multislice(self.img_data, zip(slice_dimx, slicex)).squeeze()
                        image = PIL.Image.fromarray(np.flip(np.swapaxes(image, 0, 1), 0), "RGB")

                        # Generate image suffix
                        filename_suffix_text = ''
                        for dim, slice in zip(slice_dimx, slicex):
                            filename_suffix_text += '_dim_{dim}_slice_{slice}'.format(dim=dim, slice=slice)

                        # Save asset in appropriate format
                        filename_suffix_text += '.{fmt}'.format(fmt=exploded_format)
                        if filename_suffix_text.startswith('_'):
                            filename_suffix_text = filename_suffix_text[1:]
                        filename = os.path.join(
                            'view',
                            filename_suffix_text
                        )

                        utils.save_asset(
                            image,
                            exploded_format,
                            filename,
                            optimize=self.optimize, tile_size=self.tile_size,
                            tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                            image_quality=self.image_quality, resize_filter=self.resize_filter
                        )
            else:
                # Explode deep-zoom assets by slicing in all but 2 dimensions (2D)
                img_data_dimx = range(len(self.img_data.shape))
                for slice_dimx in itertools.combinations(img_data_dimx, self.num_dimensions - 2):
                    for slicex in utils.multiradix_recursive(np.array(self.img_data.shape)[list(slice_dimx)]):
                        # Generate image object
                        image = utils.multislice(self.img_data, zip(slice_dimx, slicex))
                        image = PIL.Image.fromarray(np.flip(np.swapaxes(image.squeeze(), 0, 1), 0))

                        # Generate image suffix
                        filename_suffix_text = ''
                        for dim, slice in zip(slice_dimx, slicex):
                            filename_suffix_text += '_dim_{dim}_slice_{slice}'.format(dim=dim, slice=slice)

                        # Save asset in appropriate format
                        filename_suffix_text += '.{fmt}'.format(fmt=exploded_format)
                        if filename_suffix_text.startswith('_'):
                            filename_suffix_text = filename_suffix_text[1:]
                        filename = os.path.join(
                            'view',
                            filename_suffix_text
                        )
                        utils.save_asset(
                            image,
                            exploded_format,
                            filename,
                            optimize=self.optimize, tile_size=self.tile_size,
                            tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                            image_quality=self.image_quality, resize_filter=self.resize_filter
                        )
        self.view_path = self.file_path  # Setting to original source NIFTI file
        return


class NIFTIProcessor(BaseRadiologyImageProcessor):
    required_inputs = ['file']

    def __init__(self, *args, **kwargs):
        super(NIFTIProcessor, self).__init__(*args, **kwargs)
        self.file = self.inputs.get('file')
        self.session = boto3.session.Session()
        self.s3_client = self.session.client('s3', config=Config(signature_version='s3v4'))
        self.upload_key = None

        try:
            self.optimize = utils.str2bool(self.inputs.get('optimize_view'))
        except AttributeError:
            self.optimize = False

        try:
            self.tile_size = int(self.inputs.get('tile_size'))
        except (ValueError, KeyError, TypeError) as e:
            self.tile_size = 128

        try:
            self.tile_overlap = int(self.inputs.get('tile_overlap'))
        except (ValueError, KeyError, TypeError) as e:
            self.tile_overlap = 0

        try:
            self.tile_format = self.inputs.get('tile_format')
            if self.tile_format is None:
                self.tile_format = "png"
        except KeyError:
            self.tile_format = "png"

        try:
            self.image_quality = float(self.inputs.get('image_quality'))
        except (ValueError, KeyError, TypeError) as e:
            self.image_quality = 1.0

        try:
            self.resize_filter = self.inputs.get('resize_filter')
        except KeyError:
            self.resize_filter = "bicubic"

    def load_and_save(self):
        if os.path.isfile(self.file):
            output_file = NIFTIOutputFile(optimize=self.optimize, tile_size=self.tile_size,
                                          tile_overlap=self.tile_overlap, tile_format=self.tile_format,
                                          image_quality=self.image_quality, resize_filter=self.resize_filter)
            output_file.load_image(self.file)
            self.outputs.append(output_file)
        elif isinstance(self.file, list):
            raise NotImplementedError

    def task(self):
        # Load and save view images
        self.load_and_save()

        # Save dimensions object as JSON in view/ directory (for now)
        for dim_key in self.outputs[0].img_dimensions['dimensions'].keys():
            with open(os.path.join('view', 'dimensions%i.json' % dim_key), 'w') as fp:
                json.dump(self.outputs[0].img_dimensions['dimensions'][dim_key], fp)

        # Create create-asset JSON object file called view_asset_info.json
        self.upload_key = os.path.join(
            self.settings.storage_directory,
            'view'
        )
        with open('view_asset_info.json', 'w') as fp:
            json.dump(self.outputs[0].get_view_asset_dict(
                self.settings.storage_bucket,
                self.upload_key
            ),
                fp)

        # Upload output file to s3
        self.upload_key = os.path.join(
            self.settings.storage_directory,
            os.path.basename(self.outputs[0].file_path)
        )
        self._upload(self.outputs[0].file_path, self.upload_key)

        # Upload view assets to s3
        for view_dir_root, view_dir_folders, view_assets in os.walk(
                'view'):
            if not view_assets:
                continue
            for view_asset in view_assets:
                key_suffix = os.path.join(
                    view_dir_root,
                    view_asset
                )
                self.upload_key = os.path.join(
                    self.settings.storage_directory,
                    key_suffix
                )
                self._upload(key_suffix, self.upload_key)
