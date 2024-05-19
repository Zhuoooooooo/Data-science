import torch, pandas as pd, numpy as np
pd.set_option('display.max_columns', 12)

df = pd.read_csv('kaggle_titanic/train.csv')
df.head()