FROM nvcr.io/nvidia/tensorrt:24.04-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip pip install torchvision torchaudio
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip pip install huggingface_hub[cli]
RUN --mount=type=cache,target=/root/.cache/pip pip install gradio

RUN mkdir /app
RUN mkdir /app/pretrained_weights
WORKDIR /app/pretrained_weights
RUN huggingface-cli download --resume-download stabilityai/stable-video-diffusion-img2vid-xt --local-dir . --include "*.json"
RUN wget -c https://huggingface.co/LeonJoe13/Sonic/resolve/main/yoloface_v5m.pt
RUN wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -P vae
RUN wget -c https://huggingface.co/FoivosPar/Arc2Face/resolve/da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx
RUN huggingface-cli download --resume-download tencent/HunyuanPortrait --local-dir hyportrait

WORKDIR /app
COPY . /app

EXPOSE 8089
CMD [ "python", "gradio_app.py" ]

