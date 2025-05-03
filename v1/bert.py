import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import pickle

# モデルとトークナイザーの準備
model = AutoModelForSequenceClassification.from_pretrained("your-hub-repo-or-model")
tokenizer = AutoTokenizer.from_pretrained("your-hub-repo-or-model")

# モデル保存 (safe_serialization=True による safetensors 形式での保存)
model.save_pretrained("./role_classifier_model", safe_serialization=True)
tokenizer.save_pretrained("./role_classifier_model", safe_serialization=True)

# モデル＆トークナイザーの読み込み
model_dir = "./role_classifier_model"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ラベルエンコーダーの読み込み
with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# テスト用の発言
test_texts = [
    "それでは次の議題に移りましょう。",
    "新しいアイデアなんですが、こうしてみたらどう？",
    "あと5分で終了時間になりますのでご注意ください。",
    "今の意見をまとめると、A案が有力だと思います。",
    "この日程で皆さん大丈夫でしょうか？調整も可能です。"
]

# 推論
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # GPU対応
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_label_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label_idx].item()
    pred_label = label_encoder.inverse_transform([pred_label_idx])[0]
    print(f"発言: {text}\n→ 予測ラベル: {pred_label}, 信頼度: {confidence:.2f}\n")