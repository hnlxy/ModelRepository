# 基于ultralytics镜像构建PointNet2环境
FROM ultralytics/ultralytics:latest

# 设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PYTHONPATH="${PYTHONPATH}:/workspace:/workspace/algorithm"
ENV CUDA_VISIBLE_DEVICES=0

# 更新包管理器并安装编译依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    gcc \
    g++ \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 安装基础Python依赖
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 安装PointNet2相关依赖
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    tqdm==4.65.0 \
    h5py==3.8.0 \
    scikit-learn==1.2.2 \
    plyfile==0.7.4 \
    open3d==0.18.0 \
    pyyaml==6.0

# 单独安装点云处理相关库，添加错误处理
RUN pip install --no-cache-dir pointnet2-ops || echo "pointnet2-ops installation failed, continuing..."
RUN pip install --no-cache-dir pytorch3d || echo "pytorch3d installation failed, continuing..."

# 创建必要的目录
RUN mkdir -p /workspace/dataset/data \
    /workspace/input \
    /workspace/output/log/classification \
    /workspace/output/models \
    /workspace/output/results \
    /workspace/output/evaluation \
    /workspace/output/visualization

# 复制代码到容器
COPY . /workspace/

# 设置权限
RUN chmod +x /workspace/train.sh \
    /workspace/test.sh \
    /workspace/inference.sh

# 默认启动命令
CMD ["/bin/bash"]
