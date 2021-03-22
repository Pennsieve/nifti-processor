# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEST IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM pennsieve/base-processor-pandas-test:6-43b7408 as test

RUN apk add --no-cache --update gcc musl-dev libffi-dev jpeg-dev zlib-dev
RUN pip install --no-cache-dir moto pylint pytest pytest-cov

# install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir nibabel \
                               pillow

# code: processor
COPY base_image_radiology_processor     /app/base_image_radiology_processor
COPY nifti_processor/nifti_processor    /app/nifti_processor
COPY nifti_processor/run.py             /app/run.py

# code: tests
COPY nifti_processor/tests              /app/tests

WORKDIR /app

ENTRYPOINT [""]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PRODUCTION IMAGE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FROM pennsieve/base-processor-pandas:6-43b7408 as prod

RUN apk add --no-cache --update gcc musl-dev libffi-dev openssl-dev jpeg-dev zlib-dev
RUN pip install --no-cache-dir moto pylint pytest pytest-cov

# install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir nibabel \
                               pillow

# code: processor
COPY base_image_radiology_processor     /app/base_image_radiology_processor
COPY nifti_processor/nifti_processor    /app/nifti_processor
COPY nifti_processor/run.py             /app/run.py

# code: tests
COPY nifti_processor/tests              /app/tests

WORKDIR /app

ENTRYPOINT [""]
