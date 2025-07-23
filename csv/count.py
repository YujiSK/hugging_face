import pandas as pd

def reorder_csv(file_path):
  # CSVファイルを読み込み
  df = pd.read_csv(file_path)

  # 役割ラベルの順序を定義（各サイクルでの順番）
  role_order = [
    "ファシリテーター",
    "タイムキーパー",
    "アイデアマン",
    "クリティカルシンカー",
    "リスナー",
    "書記",
    "調整役"
  ]

  # 各役割ごとにデータをグループ化（コピーを取っておく）
  groups = {role: df[df["役割ラベル"] == role].copy() for role in role_order}

  # 各グループから1行ずつ抜き出して、role_orderの順に並べる
  ordered_dfs = []
  while any(not groups[r].empty for r in role_order):
    cycle = [groups[r].iloc[[0]] for r in role_order if not groups[r].empty]
    [groups.__setitem__(r, groups[r].iloc[1:]) for r in role_order if not groups[r].empty]
    ordered_dfs.append(pd.concat(cycle))

  # すべてのサイクルを結合して、並べ替えたデータフレームにする
  ordered_df = pd.concat(ordered_dfs, ignore_index=True)

  # 結果をCSVファイルに上書き保存
  ordered_df.to_csv(file_path, index=False)
  print(f"並べ替えたCSVが{file_path}に保存されました")
# end def

def show_label_counts(file_path):
  df = pd.read_csv(file_path)
  label_counts = df["役割ラベル"].value_counts()
  print("役割ラベルのカウント:")
  print(label_counts)  
# end def

if __name__ == '__main__':
  # csv_file = "role_classification.csv"
  csv_file = "role_classification_additional.csv"
  # reorder_csv(csv_file)
  show_label_counts(csv_file)
# end if