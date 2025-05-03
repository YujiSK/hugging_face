from transformers import BartTokenizer, BartForSequenceClassification
import torch
import pickle

# 1️⃣ モデルとトークナイザーのパス
model_dir = "./role_classifier_model"

# 2️⃣ モデル・トークナイザー・ラベルエンコーダーをBARTで！
tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForSequenceClassification.from_pretrained(model_dir)

with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 3️⃣ 分類したいテキスト（複数OK）
texts = [
    "それでは次の議題に移りたいと思います。時間も限られていますので、効率よく進めましょう。",
    "あと5分で終了時間となりますので、ご確認ください。必要に応じて延長も検討します。",
    "新しいアイデアとして、AIを活用した提案もいくつか出ています。さらに深掘りしていきたいです。"
]

# 4️⃣ 推論処理
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_label_id = torch.argmax(probs, dim=1).item()
        pred_label = label_encoder.inverse_transform([pred_label_id])[0]
        confidence = probs[0][pred_label_id].item()
    
    print(f"発言: {text} → 役割: {pred_label} (信頼度: {confidence:.2f})")
