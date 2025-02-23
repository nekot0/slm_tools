# Llama-3.1-Swallow-8B-Instruct-v0.3 推論・LoRA チューニング環境

このリポジトリには、Llama-3.1-Swallow-8B-Instruct-v0.3 の **推論・LoRA チューニング** を行うためのスクリプトが含まれています。

## 📌 機能
- **Hugging Face からモデルをダウンロード (`save_model.py`)**
- **推論 (`inference.py`)** : 指定した LLM モデルでテキスト生成を行う
- **LoRA チューニング (`lora.py`, `run_single_lora.sh`)** : LoRA (Low-Rank Adaptation) を使用してモデルの微調整を行う
- **LoRA アダプター統合 (`merge_adapter.py`, `run_merge_adapter.sh`)** : LoRA で学習したアダプターをベースモデルに統合


---

## 🛠️ 環境構築手順

### **1. リポジトリをクローン**
```bash
git clone https://github.com/nekot0/slm_tools.git
cd slm_tools/llama-3.1-swallow-tools
```

### **2. 必要なモデルをダウンロード**
```bash
python save_model.py
```
このスクリプトにより、Hugging Face から Llama-3.1-Swallow-8B-Instruct-v0.3 のモデルデータが `/data/models/Llama-3.1-Swallow-8B-Instruct-v0.3` に保存されます。

---

## 📂 ファイル構成
```
llama-lora-tuning/
├── inference.py            # 推論用スクリプト
├── lora.py                 # LoRA チューニング用スクリプト
├── merge_adapter.py        # LoRA アダプター統合スクリプト
├── save_model.py           # Hugging Face からモデルをダウンロード
├── run_single_lora.sh      # LoRA チューニング実行スクリプト
├── run_merge_adapter.sh    # LoRA 統合実行スクリプト
└── README.md               # このドキュメント
```

---

## 📍 モデルと学習データの配置
- **モデルの配置:** `/data/models/Llama-3.1-Swallow-8B-Instruct-v0.3` にダウンロード
- **LoRA 学習済みアダプターの保存:** `/data/models/adapter`
- **統合後のモデルの保存:** `/data/models/merged_model`
- **学習データ (JSONL) の保存:** `training_data.jsonl`

---

## 📝 学習データ (JSONL) の記述方法
LoRA チューニングのための学習データは **JSON Lines (JSONL) 形式** で記述します。

### **学習データのフォーマット例 (`training_data.jsonl`)**
```jsonl
{"prompt": "京アニの代表作は？", "response": "涼宮ハルヒの憂鬱、けいおん！、ヴァイオレット・エヴァーガーデンなどがあります。"}
{"prompt": "日本のAI技術の現状は？", "response": "日本のAI技術は自然言語処理や自動運転などの分野で進展しています。"}
```

---

## 🚀 推論の実行
```bash
python inference.py \
    --model_path /data/models/Llama-3.1-Swallow-8B-Instruct-v0.3 \
    --prompt "京アニの代表作は？"
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
本リポジトリのスクリプト (`inference.py`, `lora.py`, `merge_adapter.py` など) は **MIT License** のもとで提供されます。
ただし、本リポジトリで使用する **Llama-3.1-Swallow-8B-Instruct-v0.3** や依存ライブラリは、それぞれの提供元のライセンスに従います。

### ⚠️ 重要な注意点
- **Llama-3.1-Swallow-8B-Instruct-v0.3 のライセンス**
  本モデルは TokyoTech-LLM チームが提供しており、Meta の Llama 3 に基づいています。
  使用する際は、[Meta Llama 3 License](https://ai.meta.com/llama/) を確認してください。
  **商用利用が制限される可能性があります。**
- **依存ライブラリのライセンス**
  本リポジトリで使用する `transformers`, `peft`, `torch` などのライブラリは Apache License 2.0, BSD などのライセンスで提供されています。
  詳細は各ライブラリの公式ドキュメントを参照してください。

本リポジトリの **MIT License は、これらの依存ライブラリおよび Llama モデルには適用されません**。
使用する際は、各ライセンスの条件を確認してください。

