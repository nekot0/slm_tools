# Gemma-2-9B-IT 推論・LoRA チューニング環境

このリポジトリには、**Google の Gemma-2-9B-IT** の **推論・LoRA チューニング** を行うためのスクリプトが含まれています。

## 📌 機能
- **モデルのダウンロード (`save_model.py`)**: Hugging Face から Gemma-2-9B-IT を取得
- **推論 (`inference.py`)**: 指定した LLM モデルでテキスト生成を行う
- **LoRA チューニング (`lora.py`, `run_single_lora.sh`)**: LoRA (Low-Rank Adaptation) を使用してモデルの微調整を行う
- **LoRA アダプター統合 (`merge_adapter.py`, `run_merge_adapter.sh`)**: LoRA で学習したアダプターをベースモデルに統合

---

## 🛠️ 環境構築手順

### **1. リポジトリをクローン**
```bash
git clone https://github.com/nekot0/slm_tools.git
cd slm_tools/gemma-2-9b-tools
```

### **2. 必要なモデルをダウンロード**
Hugging Face のアクセストークンを使ってログインし、モデルを取得する必要があります。

```bash
huggingface-cli login
```
その後、以下のスクリプトを実行し、Gemma-2-9B-IT をダウンロードします。

```bash
python save_model.py
```
このスクリプトにより、Hugging Face から **Gemma-2-9B-IT** のモデルデータが `/data/models/gemma-2-9b-it` に保存されます。

---

## 📂 ファイル構成
```
gemma-lora-tuning/
├── save_model.py           # Hugging Face からモデルをダウンロード
├── inference.py            # 推論用スクリプト
├── lora.py                 # LoRA チューニング用スクリプト
├── merge_adapter.py        # LoRA アダプター統合スクリプト
├── run_single_lora.sh      # LoRA チューニング実行スクリプト
├── run_merge_adapter.sh    # LoRA 統合実行スクリプト
└── README.md               # このドキュメント
```

---

## 📍 モデルと学習データの配置
- **モデルの配置:** `/data/models/gemma-2-9b-it` にダウンロード
- **LoRA 学習済みアダプターの保存:** `/data/models/adapter`
- **統合後のモデルの保存:** `/data/models/merged_model`
- **学習データ (JSONL) の保存:** `training_data.jsonl`

---

## 📝 学習データ (JSONL) の記述方法
LoRA チューニングのための学習データは **JSON Lines (JSONL) 形式** で記述します。

### **学習データのフォーマット例 (`training_data.jsonl`)**
```jsonl
{"prompt": "LLMとは何ですか？", "response": "LLM（大規模言語モデル）は、大量のデータを学習し、テキスト生成などを行うAIです。"}
{"prompt": "Google の Gemma モデルについて教えてください。", "response": "Gemma は Google が開発した軽量な LLM で、オープンソースとして提供されています。"}
```

---

## 🚀 推論の実行
```bash
python inference.py \
    --model_path /data/models/gemma-2-9b-it \
    --prompt "LLMとはなんですか？"
```

---

## 🔧 LoRA チューニングの実行
### **1. チューニング実行**
```bash
bash run_single_lora.sh
```

### **2. LoRA アダプターの統合**
```bash
bash run_merge_adapter.sh
```

---

## 📜 ライセンス
本リポジトリのスクリプト (`save_model.py`, `inference.py`, `lora.py` など) は **MIT License** のもとで提供されます。
ただし、本リポジトリで使用する **Gemma-2-9B-IT モデルや依存ライブラリ** は、それぞれの提供元のライセンスに従います。

### ⚠️ 重要な注意点
- **Gemma-2-9B-IT のライセンス**
  本モデルは **Google によって提供** されており、使用する際は [Gemma の公式ライセンス](https://ai.google.dev/gemma) を確認してください。
  **商用利用の可否などに注意が必要です。**
- **依存ライブラリのライセンス**
  - `transformers`, `peft`, `torch` などのライブラリは **Apache License 2.0, BSD** などのライセンスで提供されています。
  - 詳細は各ライブラリの公式ドキュメントを参照してください。

本リポジトリの **MIT License は、これらの依存ライブラリおよび Gemma モデルには適用されません**。
使用する際は、各ライセンスの条件を確認してください。

