from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os, zipfile, pickle
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

# --- 認証 ---
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

print("📦 Google Drive から model_backup.zip を探します...")

# フォルダID（共有フォルダのURL末尾）を指定
folder_id = "1-641qI1N4zclNcUtiC25S_17jV4L6Nc2"

# Drive内で model_backup.zip を検索
file_list = drive.ListFile({
    'q': f"'{folder_id}' in parents and title = 'model_backup.zip'"
}).GetList()

if not file_list:
    raise FileNotFoundError("model_backup.zip が見つかりません。")

file_id = file_list[0]['id']
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile("model_backup.zip")
print("✅ model_backup.zip をダウンロードしました。")

# --- 解凍 ---
extract_path = "role_classifier_model"
with zipfile.ZipFile("model_backup.zip", 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(f"✅ {extract_path} に解凍しました。")

# --- モデル・トークナイザー・ラベルエンコーダー復元 ---
model = BertForSequenceClassification.from_pretrained(extract_path)
tokenizer = BertJapaneseTokenizer.from_pretrained(extract_path)

with open(os.path.join(extract_path, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
print("✅ モデル・トークナイザー・ラベルエンコーダーの読み込み完了")

# --- 推論テスト ---
text = "それでは次の議題に移りましょうか？"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=1)
probs = torch.softmax(outputs.logits, dim=1)
confidence = probs[0, pred].item()
label = label_encoder.inverse_transform(pred.cpu().numpy())

print(f"\n▶️ 発言: {text}")
print(f"→ 予測ラベル: {label[0]}, 信頼度: {confidence:.2f}")