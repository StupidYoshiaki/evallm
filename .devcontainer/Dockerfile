# 1) ベースイメージ
ARG CUDA_IMAGE="12.4.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 2) llama.cpp ビルド用ディレクトリ
WORKDIR /opt/llama

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git build-essential cmake libcurl4-openssl-dev ocl-icd-opencl-dev opencl-headers \
    libclblast-dev libopenblas-dev \
    tmux tree curl jq\
 && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CUDA_ARCH="75"
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
 && mkdir build && cd build \
 && cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=on \
      -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CUDA_ARCH} \
      -DLLAMA_ENABLE_HTTP=on \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
 && make -j$(nproc) llama-server \
 && cp bin/llama-server /usr/local/bin/llama-server

# 3) Python 環境準備
WORKDIR /workspace
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 python3.10-venv python3-pip \
 && rm -rf /var/lib/apt/lists/* \
 # python→python3.10, pip→pip3 に紐づけ
 && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3     1

COPY requirements.txt /workspace/
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt


ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EXPOSE 8000
CMD ["/bin/bash"]

# # イメージをローカルにビルド（初回だけ）
# docker build -t evallm:latest .
# 
# # ソースをマウントして起動（以降は秒でコンテナに入れる）
# docker run --gpus all -it \
#   -p 127.0.0.1:8003:8000 \
#   -v $(pwd):/workspace \
#   evallm:latest
