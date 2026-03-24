

import torch
import ast
import pandas as pd



def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def parse_pred_column(cell, top_n=5):
    # expected format: "[('word1', score1), ('word2', score2), ...]"
    items = ast.literal_eval(cell)
    return [w for w, _ in items[:top_n]]

def build_texts_targets(df, start, end, pred_col, top_n=5):
    subset = df.iloc[start:end].copy()
    texts = subset["maskedSentence"].tolist()
    targets = [parse_pred_column(x, top_n=top_n) for x in subset[pred_col].tolist()]
    return texts, targets

def result_as_dataframe(results, target_list):
    dfs = []
    for i,c in enumerate(results):
      for target in target_list:
        #print(c)
        if target in c:
          for rows in c[target]['word_attributions']:
            _df = pd.DataFrame.from_records([rows])
            _df['Target'] = target
            _df['id'] = i
            dfs.append(_df)

    df = pd.concat(dfs, ignore_index=True)
    df.rename(columns={0: 'Token', 1: 'Score'}, inplace=True) 
    return df