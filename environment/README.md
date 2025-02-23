# SLM 検証用環境構築スクリプト

このリポジトリには、SLM（Small Language Model）の検証環境を構築するための **Docker コンテナ環境** を提供するスクリプトが含まれています。

## 📌 機能概要
- **Docker を利用した開発環境の構築**（CUDA 12.8 + cuDNN + Ubuntu 22.04）
- **NVIDIA GPU に対応したコンテナ環境**
- **LangChain、PyTorch、Transformer、FAISS などの NLP ライブラリを事前インストール**
- **日本語環境（`ja_JP.UTF-8`）のセットアップ**

---

## 🛠️ 環境構築手順

### **1. リポジトリをクローン**
```bash
git clone https://github.com/nekot0/environment.git
cd environment
```

### **2. Docker イメージのビルドとコンテナの起動**
```bash
docker compose up -d --build
```

### **3. コンテナに入る**
```bash
docker exec -it dev-env /bin/bash
```

### **4. GPU が認識されているか確認**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
`True` が表示されれば、CUDA が正しく動作しています。

---

## 📂 ファイル構成
```
slm-env-setup/
├── Dockerfile              # Docker イメージ設定
├── docker-compose.yml      # コンテナ設定
├── requirements.txt        # 必要な Python パッケージ一覧
├── README.md               # このドキュメント
```

---

## 📍 インストールされる主要ライブラリ
本環境には、以下の主要なライブラリが事前インストールされます。

| ライブラリ | 説明 |
|------------|-------------------|
| PyTorch | 深層学習フレームワーク |
| Transformers | LLM（大規模言語モデル）ライブラリ |
| LangChain | LLM のオーケストレーションライブラリ |
| FAISS-GPU | 高速なベクトル検索ライブラリ |
| SudachiPy | 日本語形態素解析ライブラリ |
| Flask | 軽量な Web API フレームワーク |

その他のライブラリの詳細は `requirements.txt` を参照してください。

---

## 📜 ライセンス
本リポジトリのスクリプト (`Dockerfile`, `docker-compose.yml`, `requirements.txt` など) は **MIT License** のもとで提供されます。
ただし、本リポジトリで使用する **依存ライブラリ** は、それぞれの提供元のライセンスに従います。

### ⚠️ 重要な注意点
- **依存ライブラリのライセンス**
  本リポジトリで使用する `transformers`, `peft`, `torch` などのライブラリは Apache License 2.0, BSD などのライセンスで提供されています。
  詳細は各ライブラリの公式ドキュメントを参照してください。

本リポジトリの **MIT License は、これらの依存ライブラリには適用されません**。
使用する際は、各ライセンスの条件を確認してください。

