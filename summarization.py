from transformers import pipeline
from dotenv import load_dotenv
import os

# .envファイルを読み込む
load_dotenv()

# トークンを取得
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# 要約パイプライン（Hugging FaceクラウドAPI使用）
summarizer = pipeline(
    "summarization", 
    model="sonoisa/summarize-ja-gpt2-medium",
    use_auth_token=API_TOKEN
)

# 長文テキスト
text = "この議題については、まずA案が出されましたが、それに対してB案の方が良いのではないかという意見があり、最終的にC案にまとまりました。"

# 要約実行
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
print("要約結果:", summary[0]['summary_text'])
