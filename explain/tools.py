

import math

import torch
import ast
from tqdm import tqdm
import pandas as pd
from spacy.tokens import Doc
import spacy
from transformers import AutoTokenizer, AutoModelForMaskedLM



def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def filter_texts_targets(texts, predicted_targets, kw_filters, data_df, only_keywords=False):
    filtered_texts = []
    filtered_predicted_targets = []
    filter_identifiers = []
    for i,(text, preds) in enumerate(zip(texts, predicted_targets)):
        if set(preds).intersection(set(kw_filters)):
            filtered_texts.append(text)
            filter_identifiers.append(i)
            if only_keywords:
                preds = [p for p in preds if p in kw_filters]
                if not preds:
                    continue
            filtered_predicted_targets.append([data_df.targetExpression.iloc[i]]+preds)
    return filter_identifiers, filtered_texts, filtered_predicted_targets

def add_pseudo_perplexity_column(
    data_df,
    checkpoint,
    text_col="currentSentence",
    output_col="token_pseudo_ppl",
    device=None,
    max_length=512,
    batch_size=None,
    use_amp=True,
):
    """Add a column to `data_df` with, per row, a list of (token, perplexity)
    tuples in sentence order for each token in `text_col`, scored under a BERT
    masked LM (Salazar et al., 2019): exp(-log P(token | rest of sentence)),
    where the token itself is masked out.

    All maskings for a given sentence are scored in one batched forward pass
    (instead of one token at a time), which is the main lever for throughput on
    an M1's unified-memory GPU (MPS) or a Colab GPU alike. `device` auto-selects
    via pick_device() (mps > cuda > cpu) if not given; `batch_size` similarly
    defaults per device (large on cuda, modest on mps/cpu) but can be overridden,
    e.g. bumped up further on a roomy Colab GPU. `use_amp` enables autocast (fp16)
    on cuda/mps for extra throughput; it's a no-op on cpu.
    """
    device = device or pick_device()
    if batch_size is None:
        batch_size = {"cuda": 64, "mps": 16}.get(device, 8)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint).to(device)
    model.eval()

    # tokenizer.model_max_length is often left at a bogus huge sentinel value by
    # checkpoints that never set it, so cap max_length at what the model's
    # position embeddings can actually hold to avoid an out-of-range index error
    # on long sentences.
    model_max_length = getattr(model.config, "max_position_embeddings", max_length)
    max_length = min(max_length, model_max_length)

    mask_id = tokenizer.mask_token_id
    results = []

    amp_enabled = use_amp and device in ("cuda", "mps")
    autocast_ctx = torch.autocast(
        device_type=device, dtype=torch.float16, enabled=amp_enabled
    )
    pin = device == "cuda"

    with torch.inference_mode(), autocast_ctx:
        for text in data_df[text_col].tolist():
            encoded = tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            input_ids = encoded["input_ids"][0]
            attention_mask = encoded["attention_mask"][0]
            special_mask = tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            )
            positions = [i for i, m in enumerate(special_mask) if m == 0]

            token_ppls = []

            for start in range(0, len(positions), batch_size):
                chunk_positions = positions[start : start + batch_size]
                chunk = len(chunk_positions)

                batch_input_ids = input_ids.unsqueeze(0).repeat(chunk, 1).clone()
                batch_attention_mask = attention_mask.unsqueeze(0).repeat(chunk, 1)
                pos_tensor = torch.tensor(chunk_positions, dtype=torch.long)
                row_idx = torch.arange(chunk)
                batch_input_ids[row_idx, pos_tensor] = mask_id

                if pin:
                    batch_input_ids = batch_input_ids.pin_memory()
                    batch_attention_mask = batch_attention_mask.pin_memory()

                batch_input_ids = batch_input_ids.to(device, non_blocking=pin)
                batch_attention_mask = batch_attention_mask.to(device, non_blocking=pin)
                pos_tensor = pos_tensor.to(device)
                row_idx = row_idx.to(device)

                logits = model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                ).logits  # (chunk, seq_len, vocab)

                # gather only the masked position's logits before softmax, so we
                # never materialize a (chunk, seq_len, vocab) softmax on the GPU
                target_logits = logits[row_idx, pos_tensor].float()  # (chunk, vocab)
                log_probs = torch.log_softmax(target_logits, dim=-1)

                target_ids = input_ids[torch.tensor(chunk_positions, dtype=torch.long)].to(device)
                token_log_probs = log_probs[row_idx, target_ids].cpu().tolist()

                for tid, log_p in zip(target_ids.cpu().tolist(), token_log_probs):
                    token_str = tokenizer.convert_ids_to_tokens(tid)
                    token_ppls.append((token_str, math.exp(-log_p)))

            results.append(token_ppls)

    data_df[output_col] = results
    return data_df

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
    for i,c in tqdm(enumerate(results), total=len(results)):
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




def _direct_dependency_relation_to_mask(tok, mask_tok):
    """Return only direct dependency relation to [MASK], else None."""
    if tok.i == mask_tok.i:
        return "MASK"

    # tok directly depends on [MASK]
    if tok.head.i == mask_tok.i:
        return f"{tok.dep_}^"

    # [MASK] directly depends on tok
    if mask_tok.head.i == tok.i:
        return f"{mask_tok.dep_}v"

    return None


def _dependency_relation_between_tokens(source_tok, target_tok):
    """Return dependency-path relation from source token to target token."""

    if source_tok.i == target_tok.i:
        return "SELF"

    by_index = {tok.i: tok for tok in source_tok.doc}

    def neighbors(tok):
        seen = {tok.head.i}
        out = [tok.head]
        for child in tok.children:
            if child.i not in seen:
                out.append(child)
                seen.add(child.i)
        return out

    queue = [(source_tok.i, [source_tok.i])]
    visited = {source_tok.i}
    found_path = None

    while queue:
        node_idx, path = queue.pop(0)
        if node_idx == target_tok.i:
            found_path = path
            break

        node = by_index[node_idx]
        for nxt in neighbors(node):
            if nxt.i in visited:
                continue
            visited.add(nxt.i)
            queue.append((nxt.i, path + [nxt.i]))

    if not found_path:
        return None

    path_tokens = [by_index[i] for i in found_path]
    labels = []
    for current_tok, next_tok in zip(path_tokens[:-1], path_tokens[1:]):
        if current_tok.head.i == next_tok.i:
            labels.append(f"{current_tok.dep_}^")
        elif next_tok.head.i == current_tok.i:
            labels.append(f"{next_tok.dep_}v")
        else:
            return None

    return "|".join(labels) if labels else "SELF"


def _relation_to_constituent(tok, constituent_tokens):
    """Return nearest dependency relation from token to any token in constituent."""
    if not constituent_tokens:
        return None

    best_relation = None
    best_steps = None

    for target_tok in constituent_tokens:
        relation = _dependency_relation_between_tokens(tok, target_tok)
        if relation is None:
            continue

        steps = 0 if relation == "SELF" else relation.count("|") + 1
        if best_steps is None or steps < best_steps:
            best_relation = relation
            best_steps = steps

    return best_relation


def add_mask_syntax_relation(
    df,
    token_col="Token",
    id_col="id",
    target_col="Target",
    identifier_col="identifier",
    mask_token="[MASK]",
    same_constituent_col="mask_syntax_relation",
    outside_constituent_col="mask_constituent_relation",
    spacy_model="en_core_web_sm",
):
    """Add dependency relations for tokens relative to the [MASK] constituent.

    - Tokens in the same constituent as [MASK] are labeled in same_constituent_col.
    - Tokens outside that constituent are labeled in outside_constituent_col with
      the nearest dependency relation to the [MASK] constituent.
    - If no relation can be computed, both columns are None.
    """
    if token_col not in df.columns or id_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{token_col}' and '{id_col}' columns.")

    nlp = spacy.load(spacy_model, disable=["ner"])

    group_cols = [id_col]
    if target_col in df.columns:
        group_cols.append(target_col)
    if identifier_col in df.columns:
        group_cols.append(identifier_col)

    df_out = df.copy()
    df_out[same_constituent_col] = None
    df_out[outside_constituent_col] = None
    relation_cache = {}

    grouped_indices = df_out.groupby(group_cols, sort=False).groups
    for _, idx in tqdm(grouped_indices.items(), total=len(grouped_indices), desc="Constituent relation"):
        rows = df_out.loc[idx]
        tokens = rows[token_col].astype(str).tolist()
        token_key = tuple(tokens)

        if token_key in relation_cache:
            in_relations, out_relations = relation_cache[token_key]
        else:
            mask_positions = [i for i, tok in enumerate(tokens) if tok == mask_token]
            if len(mask_positions) != 1:
                in_relations = [None] * len(tokens)
                out_relations = [None] * len(tokens)
            else:
                mask_idx = mask_positions[0]
                doc = Doc(nlp.vocab, words=tokens)
                for _, pipe in nlp.pipeline:
                    doc = pipe(doc)
                mask_tok = doc[mask_idx]
                mask_constituent_indices = {tok.i for tok in mask_tok.subtree}
                mask_constituent_tokens = [doc[i] for i in mask_constituent_indices]

                in_relations = []
                out_relations = []

                for tok in doc:
                    if tok.i in mask_constituent_indices:
                        in_relations.append(_dependency_relation_between_tokens(tok, mask_tok))
                        out_relations.append(None)
                    else:
                        in_relations.append(None)
                        out_relations.append(_relation_to_constituent(tok, mask_constituent_tokens))

            relation_cache[token_key] = (in_relations, out_relations)

        if len(in_relations) != len(rows) or len(out_relations) != len(rows):
            in_relations = [None] * len(rows)
            out_relations = [None] * len(rows)

        df_out.loc[idx, same_constituent_col] = in_relations
        df_out.loc[idx, outside_constituent_col] = out_relations

    return df_out

def add_mask_token_distance(
    df,
    token_col="Token",
    id_col="id",
    target_col="Target",
    identifier_col="identifier",
    mask_token="[MASK]",
    distance_col="mask_token_distance",
    signed=False,
    ):
    """Add token-position distance to [MASK] for each token in each sentence group.

    Distance is absolute by default. Set signed=True to keep left/right direction
    (negative: left of [MASK], positive: right of [MASK]).
    """
    if token_col not in df.columns or id_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{token_col}' and '{id_col}' columns.")

    group_cols = [id_col]
    if target_col in df.columns:
        group_cols.append(target_col)
    if identifier_col in df.columns:
        group_cols.append(identifier_col)

    df_out = df.copy()
    df_out[distance_col] = None
    distance_cache = {}

    grouped_indices = df_out.groupby(group_cols, sort=False).groups
    for _, idx in tqdm(grouped_indices.items(), total=len(grouped_indices), desc="Token distance"):
        rows = df_out.loc[idx]
        tokens = rows[token_col].astype(str).tolist()
        token_key = tuple(tokens)

        if token_key in distance_cache:
            distances = distance_cache[token_key]
        else:
            mask_positions = [i for i, tok in enumerate(tokens) if tok == mask_token]
            if len(mask_positions) != 1:
                distances = [None] * len(tokens)
            else:
                mask_idx = mask_positions[0]
                if signed:
                    distances = [i - mask_idx for i in range(len(tokens))]
                else:
                    distances = [abs(i - mask_idx) for i in range(len(tokens))]

            distance_cache[token_key] = distances

        if len(distances) != len(rows):
            distances = [None] * len(rows)

        df_out.loc[idx, distance_col] = distances

    return df_out

from typing import List, Tuple, Literal

def merge_subtokens(
    tokens: List[Tuple[str, float]],
    agg: Literal["mean", "max", "min", "sum", "first"] = "mean"
) -> List[Tuple[str, float]]:
    """
    Merge WordPiece-style subtokens (prefixed with '##') back into their
    parent token, aggregating the associated scores.

    Args:
        tokens: List of (token, score) tuples, e.g. from a tokenizer/model
                output such as [("Hello", 0.9), ("##world", 0.8), ...]
        agg: How to combine scores for merged tokens - "mean", "max",
             "min", "sum", or "first" (keep only the first subtoken's score).

    Returns:
        List of (merged_token, aggregated_score) tuples.
    """
    if not tokens:
        return []

    agg_funcs = {
        "mean": lambda scores: sum(scores) / len(scores),
        "max": max,
        "min": min,
        "sum": sum,
        "first": lambda scores: scores[0],
    }
    if agg not in agg_funcs:
        raise ValueError(f"Unknown agg method: {agg!r}")
    combine = agg_funcs[agg]

    result = []
    current_text_parts = [tokens[0][0]]
    current_scores = [tokens[0][1]]

    for token, score in tokens[1:]:
        if token.startswith("##"):
            current_text_parts.append(token[2:])
            current_scores.append(score)
        else:
            # flush the previous merged token
            result.append(("".join(current_text_parts), combine(current_scores)))
            current_text_parts = [token]
            current_scores = [score]

    # flush the last one
    result.append(("".join(current_text_parts), combine(current_scores)))

    return result