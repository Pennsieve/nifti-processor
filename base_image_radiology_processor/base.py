from base_processor.imaging import BaseImageProcessor


class BaseRadiologyImageProcessor(BaseImageProcessor):
    def __init__(self, *args, **kwargs):
        super(BaseRadiologyImageProcessor, self).__init__(*args, **kwargs)

