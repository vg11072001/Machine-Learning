import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("dataset/train.csv")
print('-----------train-----------')
print(df.head())
print(df.shape)
print(df.columns)
sgkf = StratifiedGroupKFold(n_splits=5, random_state=0, shuffle=True)
group_id = df["prompt"]
label_id = df["winner_model_a winner_model_b winner_tie".split()].values.argmax(1)
splits = list(sgkf.split(df, label_id, group_id))

df["fold"] = -1
for fold, (_, valid_idx) in enumerate(splits):
    df.loc[valid_idx, "fold"] = fold
print('-----------folds-----------')
print(df["fold"].value_counts())
print('-----------dtrainval-----------')
print(df.head())
print(df.shape)
print(df.columns)
# df.to_csv("../artifacts/dtrainval.csv", index=False)
df.to_csv("D:/Personal/ML/ml notes/Kaggle/solution-lmsys-chatbot-arena/dataset/dtrainval.csv", index=False)
