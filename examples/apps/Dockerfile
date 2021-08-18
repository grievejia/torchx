FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install classy_vision pytorch-lightning fsspec[s3] torch-model-archiver captum boto3 lightning-bolts

WORKDIR /app

COPY . /app
