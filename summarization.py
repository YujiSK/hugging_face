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
    model="tsmatz/mt5_summarize_japanese",
    token=API_TOKEN
)

# 長文テキスト
text = "本日の全社会議では、今年度の売上実績と次年度の戦略について詳細な議論が行われました。まず、財務部から売上や利益率、コスト構造に関する報告があり、全体として目標には到達しているものの、一部の製品ラインで利益率が低下していることが指摘されました。その原因としては、原材料価格の高騰や物流コストの増加が挙げられ、今後はより柔軟な調達体制の構築が求められることが確認されました。次に、マーケティング部からは今後の市場動向と競合分析について発表があり、特に新興市場への参入やデジタルマーケティング施策の強化が必要であると提案されました。また、開発部門からは新製品の開発計画について報告があり、AI技術を活用した製品ラインの拡充が進められていることが共有されました。最後に、全社員からの質疑応答が行われ、働き方改革や社内コミュニケーションの改善に関する意見も出され、経営陣はこれらの意見を今後の施策に反映させることを約束しました。"

# 要約実行
summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
print("要約結果:", summary[0]['summary_text'])
