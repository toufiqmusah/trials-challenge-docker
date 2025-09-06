FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

WORKDIR /user

COPY ./src /user 
RUN pip install -r requirements.txt

RUN pip install --upgrade pip --no-cache-dir && \
    pip install -e ./src/nn-SegMamba/nnUNet --no-cache-dir && \
    pip install -e ./src/nn-SegMamba/mednext --no-cache-dir && \
    pip install monai

ENTRYPOINT [ "/bin/bash", "/user/run.sh" ]