# 🧠 MeetingRoleClassifier-HF

A multimodal analysis tool that classifies and summarizes participant roles in meetings using both **audio and text inputs**.  
Designed as a research and educational project for analyzing facilitation dynamics and meeting structures.

---

## 🎯 Overview

- **Objective**: Automatically classify roles (e.g., Facilitator, Speaker, Notetaker) based on recorded speech and text data from meetings, and optionally summarize them.
- **Key Features**:
  - Speech-to-text conversion (WAV input)
  - CSV-based role classification and aggregation
  - Transformer-based inference using BERT/BART
  - Versioned pipelines for model testing (`v1`, `v2`)
  - Lightweight and extensible architecture

---

## 📁 Project Structure

```

MeetingRoleClassifier-HF/
├── audio/                # Audio files (.wav)
├── csv/                  # Data and preprocessing scripts
│   ├── count.py
│   ├── role\_classification.csv
│   └── role\_classification\_additional.csv
├── results/              # Inference or aggregation output
├── v1/                   # BERT-based role classification
├── v2/                   # Extended model (e.g., BART summarization)
├── README.md

````

---

## 🛠 Tech Stack

| Category | Tools |
|---------|-------|
| Language | Python 3.x |
| Models | BERT / BART (exact variants TBD) |
| Speech Recognition | `speech_to_text.py` (model TBD) |
| Key Libraries | pandas, transformers, torch, scikit-learn, soundfile (assumed from imports) |

---

## 🔁 Pipeline Overview

1. **CSV Preprocessing & Aggregation**
   - Reorder role labels (`reorder_csv`)
   - Count labels and visualize statistics (`show_label_counts`)
2. **Role Classification & Summarization**
   - Load utterances + labels from CSV
   - Predict roles or summarize using transformer models
3. **Save results** in `/results`

---

## 🚀 Usage

### 1. Aggregate Labels from CSV
```bash
python csv/count.py csv/role_classification.csv
````

> Note: File names are hardcoded. Modify if needed for other CSVs.

### 2. Run Inference with BERT (v1)

```bash
cd v1
python bert.py --input ../csv/role_classification.csv --output ../results/v1_output.csv
```

> See script headers for detailed arguments.

---

## 📥 Input & Output Example

* **Input**:

  * Audio file: `audio/sample.wav`
  * Text + label CSV:

```csv
Utterance,Label
"Let's get started.",Facilitator
"I agree with that suggestion.",Participant
...
```

* **Output**:

  * Predicted role labels (CSV)
  * Optional: summary text

---

## 🔭 Planned Improvements

* [ ] Specify model versions (e.g., `bert-base-japanese`)
* [ ] Evaluate performance (Accuracy / F1-score)
* [ ] Add a simple UI (e.g., Streamlit)
* [ ] Support batch processing of multiple audio files
* [ ] Upload trained models to Hugging Face

---

## 🧪 Applications

* Meeting transcription structuring
* Facilitation pattern analysis
* Multimodal dialogue research
* Capstone / course / thesis projects

---

## 👤 My Contributions (e.g., Yuji Sunagawa)

* CSV sorting and role count aggregation (`count.py`)
* Model inference setup and validation (v1, v2)
* Project folder design and version control
* Refactoring for maintainability and error handling

---

## 🔗 Repository (tentative)

[https://github.com/YujiSK/MeetingRoleClassifier-HF](https://github.com/YujiSK/MeetingRoleClassifier-HF)

---

## 📝 License

To be released under the MIT License.

---


# 🧠 MeetingRoleClassifier-HF (日本語)

音声とテキストの両方を活用して、会議参加者の役割を分類・要約するマルチモーダル解析ツールです。  
会議の議事録作成やファシリテーション分析に役立つ、研究・教育向けプロジェクトとして設計されています。

---

## 🎯 プロジェクト概要

- **目的**：会議の録音・書き起こしデータから、参加者の役割（例：ファシリテーター・発言者・書記など）を自動分類し、必要に応じて要約も生成。
- **特徴**：
  - 音声（`.wav`） → テキスト変換
  - CSVデータ（発言＋ラベル）による分類・集計
  - BERT/BARTベースのモデルによる学習・推論
  - 学習済みモデルのバージョン管理（v1, v2）
  - シンプルな構成で拡張容易

---

## 📁 ディレクトリ構成

```
MeetingRoleClassifier-HF/
├── audio/                # 音声ファイル (.wav)
├── csv/                  # データ処理用CSVとスクリプト
│   ├── count.py
│   ├── role_classification.csv
│   └── role_classification_additional.csv
├── results/              # モデル推論や集計結果の保存先
├── v1/                   # 初期バージョン：BERTベース分類
├── v2/                   # 拡張版：BART要約モデル等
├── README.md
```

---

## 🛠 使用技術

| 分類 | 内容 |
|------|------|
| 言語 | Python 3.x |
| モデル | BERT / BART 系（後日確定） |
| 音声認識 | speech_to_text.py を使用（モデル詳細は未特定） |
| 主要ライブラリ | pandas, transformers, torch, scikit-learn, soundfile など（推定） |

---

## 🔁 主な処理フロー

1. `count.py` による前処理・集計
   - ラベル順の並び替え
   - 発言数・役割ラベルの件数集計
2. `v1`, `v2` にてモデルの推論・学習
   - CSVから発言とラベルを読み込み
   - モデルによる役割予測または要約生成
3. 結果を `results/` に保存

---

## 🖥 実行方法

### CSVの並べ替えと集計
```bash
python csv/count.py csv/role_classification.csv
```

> ※ デフォルトではファイル名がスクリプト内で指定されています。任意のCSVで実行するには適宜変更してください。

### モデル推論（例：v1/BERTモデル使用）

```bash
cd v1
python bert.py --input ../csv/role_classification.csv --output ../results/v1_output.csv
```

> 詳細なオプションは各スクリプト冒頭をご確認ください。

---

## 💡 入出力例

* **入力**：

  * 音声ファイル：`audio/sample.wav`
  * テキストデータ：`role_classification.csv`

```csv
発言,ラベル
"それでは始めましょう",ファシリテーター
"その案には賛成です",参加者
...
```

* **出力**（例）：

  * 役割分類ラベル一覧（CSV）
  * 要約テキスト（オプション）

---

## 🔬 今後の改善予定

* [ ] 使用モデルの明示（bert-base-japanese など）
* [ ] 推論精度の評価（Accuracy / F1 Score など）
* [ ] 推論UIの作成（Streamlitなど）
* [ ] 複数音声ファイルへのバッチ適用
* [ ] Hugging Faceに学習済みモデルアップロード

---

## 📚 応用・用途

* 会議録音の構造化・分類
* ファシリテーション研究支援
* 音声インターフェース付き対話分析
* 授業・研究プロジェクトとしての応用

---

## 👤 担当範囲（例：砂川優治）

* CSVデータの並べ替え・集計（`count.py`）
* モデル推論パイプラインの整備（v1, v2）
* ディレクトリ設計・バージョン管理
* コードの可読性向上とエラーハンドリング対応

---

## 📎 リンク（仮）

* GitHubリポジトリ：[https://github.com/YujiSK/MeetingRoleClassifier-HF](https://github.com/YujiSK/MeetingRoleClassifier-HF)

---

## 📝 ライセンス

MIT License（予定）

---
