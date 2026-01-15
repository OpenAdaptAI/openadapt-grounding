FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git-lfs \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Accept Conda Terms of Service (required as of late 2024)
RUN conda config --set solver libmamba && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n omni python=3.12 && \
    echo "source activate omni" > ~/.bashrc
ENV CONDA_DEFAULT_ENV=omni
ENV PATH="/opt/conda/envs/omni/bin:$PATH"

WORKDIR /app

RUN git clone https://github.com/microsoft/OmniParser.git && \
    cd OmniParser && \
    git lfs install && \
    git lfs pull

WORKDIR /app/OmniParser

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate omni && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install -r requirements.txt && \
    pip install huggingface_hub fastapi uvicorn && \
    pip install paddleocr==2.8.1 paddlepaddle==2.6.2 && \
    pip install transformers==4.44.2 && \
    pip install flash-attn --no-build-isolation

# Download V2 weights
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate omni && \
    mkdir -p /app/OmniParser/weights && \
    cd /app/OmniParser && \
    rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence && \
    for folder in icon_caption icon_detect; do \
        huggingface-cli download microsoft/OmniParser-v2.0 --local-dir weights --repo-type model --include "$folder/*"; \
    done && \
    mv weights/icon_caption weights/icon_caption_florence

# Pre-download EasyOCR model during build (PaddleOCR downloads at runtime)
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate omni && \
    cd /app/OmniParser && \
    python3 -c "import easyocr; reader = easyocr.Reader(['en']); print('Downloaded EasyOCR model')"

CMD ["python3", "/app/OmniParser/omnitool/omniparserserver/omniparserserver.py", \
     "--som_model_path", "/app/OmniParser/weights/icon_detect/model.pt", \
     "--caption_model_path", "/app/OmniParser/weights/icon_caption_florence", \
     "--device", "cuda", \
     "--BOX_TRESHOLD", "0.05", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
