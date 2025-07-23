# hugging_face_pydrive.py

# ====== PyDriveの認証と初期化 ======
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

print("Google Driveとの連携...")
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# ====== CSVファイルの検索とダウンロード ======
import pandas as pd
print("\nrole_classification.csv を探索...")

file_list = drive.ListFile({'q': "title='role_classification.csv'"}).GetList()
if not file_list:
    raise FileNotFoundError("role_classification.csv が Google Drive に見つかりません")

file_id = file_list[0]['id']
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('csv/role_classification.csv')
df = pd.read_csv('csv/role_classification.csv')

# ====== ラベルエンコード & Dataset 変換 ======
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["役割ラベル"])
dataset = Dataset.from_pandas(df[["発言内容", "label"]])

# ====== BERT Tokenizer と Model ======
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-v3", num_labels=len(label_encoder.classes_))

def tokenize_function(examples):
    return tokenizer(examples["発言内容"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ====== Trainer ======
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

trainer.train()

# ====== モデルの保存 (zip + Google Drive アップロード) ======
import shutil, pickle

save_path = "role_classifier_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
with open(f"{save_path}/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

shutil.make_archive("model_backup", 'zip', save_path)
uploaded_model = drive.CreateFile({'title': 'model_backup.zip'})
uploaded_model.SetContentFile("model_backup.zip")
uploaded_model.Upload()

print("\n🚀 モデルの訓練およびGoogle Driveへの保存が終了しました。")
