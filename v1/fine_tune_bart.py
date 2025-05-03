from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, Dataset
import pandas as pd
import os


import torch

if torch.cuda.is_available():
    print(f"✅ GPU名: {torch.cuda.get_device_name(0)}")
    print(f"🧠 メモリ総量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
else:
    print("❌ GPUは使えません（CPUのみ）")

    

# # 1 データ読み込み
# df = pd.read_csv("csv/role_classification.csv")
# print(df.head())

# # 2 ラベルを数値化
# label_encoder = LabelEncoder()
# df["label"] = label_encoder.fit_transform(df["役割ラベル"])
# print(label_encoder.classes_)  # ラベル名確認

# # 3 Hugging Face Datasetに変換
# dataset = Dataset.from_pandas(df[["発言内容", "label"]])

# # 4 トークナイザー
# tokenizer = AutoTokenizer.from_pretrained("ku-nlp/bart-large-japanese")

# # 5 データのトークン化
# def tokenize_function(examples):
#     return tokenizer(examples["発言内容"], padding="max_length", truncation=True, max_length=512)

# tokenized_dataset = dataset.map(tokenize_function, batched=True)

# # 6 モデル準備（学習済みがあればそれを使う）
# model_dir = "./role_classifier_model"
# if os.path.exists(model_dir):
#     print("学習済みモデルをロードします...")
#     model = AutoModelForSequenceClassification.from_pretrained(model_dir)
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
#     # ラベルエンコーダーもロード
#     import pickle
#     with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
#         label_encoder = pickle.load(f)
# else:
#     print("新規でモデルを作成します...")
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "ku-nlp/bart-large-japanese",
#         num_labels=len(label_encoder.classes_)
#     )
#     tokenizer = AutoTokenizer.from_pretrained("ku-nlp/bart-large-japanese")

# # 7 訓練パラメータ
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=10,
# )

# # 8 Trainerで学習
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     eval_dataset=tokenized_dataset,
# )

# trainer.train()

# # 9 モデル保存
# model.save_pretrained("./role_classifier_model", safe_serialization=False)
# tokenizer.save_pretrained("./role_classifier_model")

# # 10 ラベルも保存
# import pickle
# with open("./role_classifier_model/label_encoder.pkl", "wb") as f:
#     pickle.dump(label_encoder, f)

# print("ファインチューニング完了！")