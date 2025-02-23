# NVIDIA GPU を利用するための CUDA ベースイメージ
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

# 必要なパッケージをインストールし、キャッシュを削除して軽量化
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        language-pack-ja \
        fonts-noto-cjk \
        vim curl git wget \
        python3.10 python3.10-venv python3.10-dev python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 最新の pip をインストール
RUN pip install --upgrade pip

# 日本語ロケールの設定
RUN locale-gen ja_JP.UTF-8 && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python3.10 をデフォルトの python に設定
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 作業ディレクトリを作成・設定
WORKDIR /app

# requirements.txt をコンテナ内にコピー
COPY requirements.txt /app/

# 仮想環境を作成し、Python パッケージをインストール
RUN python3.10 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# デフォルトのシェルを bash に設定
CMD ["/bin/bash"]