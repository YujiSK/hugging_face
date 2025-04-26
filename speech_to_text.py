from transformers import pipeline
from dotenv import load_dotenv
import os

# .envファイルを読み込む
load_dotenv()

# トークンを取得
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# 音声認識パイプライン（Whisperモデル）
asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small", 
    use_auth_token=API_TOKEN
)

# 音声ファイル（例: sample.wav）
audio_file = "sample.wav"

# 音声認識実行
result = asr(audio_file)
print("認識結果:", result['text'])
