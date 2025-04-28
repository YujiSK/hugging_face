from transformers import pipeline
from dotenv import load_dotenv
import os

# .envファイルを読み込む
load_dotenv()

# トークンを取得
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# 分類パイプライン（クラウドAPI使用、BERTモデル）
classifier = pipeline(
    "text-classification", 
    model="cl-tohoku/bert-base-japanese", 
    token=API_TOKEN
)

# 発言テキスト
text = "それでは次の議題に移りましょう。"

# 分類実行
result = classifier(text)
print("分類結果:", result)
