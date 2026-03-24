

import torch



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