FROM nvcr.io/nvidia/merlin/merlin-pytorch-training:0.5.3

ADD ./sigir_ecom_transformers/requirements.txt requirements.txt

#Installing framework dependencies
RUN pip install -r requirements.txt;

#Install NVTABULAR latest version (with the new PyT data loader)
RUN cd /nvtabular && git checkout main && git pull origin main && pip install -e .
