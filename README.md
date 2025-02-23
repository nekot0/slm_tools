# Small Language Models環境構築・推論・LoRA チューニングツール

このリポジトリは、Small Language Modelsの環境構築、推論、および LoRA チューニングを行うためのスクリプトを提供します。

## 📌 概要
本リポジトリは、以下の2つのディレクトリに分かれています。

1. **`environ/`** - 推論・LoRA チューニングを行うための開発環境のセットアップスクリプト
2. **`llama-3.1-swallow-tools/`** - Llama-3.1-Swallow-8B-Instruct の推論および LoRA チューニングのためのスクリプト

各ディレクトリの詳細については、それぞれの `README.md` を参照してください。

---

## 📂 ディレクトリ構成
```
slm_tools/
├── environ/                      # 環境構築用スクリプト
│   ├── Dockerfile               # Docker イメージ設定
│   ├── docker-compose.yml       # コンテナ管理設定
│   ├── requirements.txt         # 必要な Python パッケージ一覧
│   ├── README.md                # 環境構築に関する詳細説明
│
├── llama-3.1-swallow-tools/     # Llama-3.1-Swallow-8B-Instruct 用スクリプト
│   ├── inference.py             # 推論用スクリプト
│   ├── lora.py                  # LoRA チューニング用スクリプト
│   ├── merge_adapter.py         # LoRA アダプター統合スクリプト
│   ├── save_model.py            # Hugging Face からモデルをダウンロード
│   ├── run_single_lora.sh       # LoRA チューニング実行スクリプト
│   ├── run_merge_adapter.sh     # LoRA 統合実行スクリプト
│   ├── README.md                # 推論・LoRA チューニングに関する詳細説明
│
└── README.md                    # このドキュメント
```

---

## 🚀 環境構築と利用方法

### **1. リポジトリのクローン**
```bash
git clone https://github.com/nekot0/slm_tools.git
cd slm_tools
```

### **2. 開発環境のセットアップ**
環境構築用のスクリプトは `environ/` ディレクトリに格納されています。
詳細は `environ/README.md` を参照してください。

```bash
cd environ
bash setup.sh  # または `docker compose up -d --build`
```

### **3. Llama-3.1-Swallow-8B-Instruct の推論 & LoRA チューニング**
Llama の推論および LoRA チューニングに関するスクリプトは `llama-3.1-swallow-tools/` に格納されています。
詳細は `llama-3.1-swallow-tools/README.md` を参照してください。

```bash
cd llama-3.1-swallow-tools
python inference.py --model_path /data/models/Llama-3.1-Swallow-8B-Instruct-v0.3 --prompt "京アニの代表作は？"
```

---

## 📜 ライセンス
本リポジトリのスクリプトは **MIT License** のもとで提供されます。
ただし、本リポジトリで使用する **Llama-3.1-Swallow-8B-Instruct-v0.3** や依存ライブラリは、それぞれの提供元のライセンスに従います。

### ⚠️ 重要な注意点
- **Llama-3.1-Swallow-8B-Instruct-v0.3 のライセンス**
  - 本モデルは TokyoTech-LLM チームが提供しており、Meta の Llama 3 に基づいています。
  - 使用する際は、[Meta Llama 3 License](https://ai.meta.com/llama/) を確認してください。
  - **商用利用が制限される可能性があります。**
- **依存ライブラリのライセンス**
  - 本リポジトリで使用する `transformers`, `peft`, `torch` などのライブラリは Apache License 2.0, BSD などのライセンスで提供されています。
  - 詳細は各ライブラリの公式ドキュメントを参照してください。

本リポジトリの **MIT License は、これらの依存ライブラリおよび Llama モデルには適用されません**。
使用する際は、各ライセンスの条件を確認してください。

