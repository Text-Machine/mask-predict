import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel, AutoTokenizer


def summarize_top_predictors(results, target, top_n=10):
    """
    Aggregate attributions across all sentences for a single target word.
    
    Args:
        results: output from explainer.explain()
        target: target word to summarize
        top_n: number of top predictors to return
    
    Returns:
        list of (word, mean_attribution, std_attribution, count)
    """
    word_scores = {}
    
    for sent_result in results:
        if target not in sent_result or sent_result[target].get("skipped"):
            continue
        
        word_attrs = sent_result[target]["word_attributions"]
        for word, score in word_attrs:
            if word not in word_scores:
                word_scores[word] = []
            word_scores[word].append(score)
    
    # Compute statistics
    stats = []
    for word, scores in word_scores.items():
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        count = len(scores)
        stats.append((word, mean_score, std_score, count))
    
    # Sort by mean attribution (descending)
    stats.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return stats[:top_n]


def analyze_comparison(comparison, target, top_n=15):
    """
    Summarize differences between two models across all sentences.
    Returns:
        list of (word, model1_score, model2_score, difference, std_difference)
    """
    word_diffs = {}

    for sent_comp in comparison:
        # sent_comp should be dict: {target: rows_or_skipdict}
        if not isinstance(sent_comp, dict) or target not in sent_comp:
            continue

        entry = sent_comp[target]

        # Skip record format: {"skipped": True, ...}
        if isinstance(entry, dict):
            if entry.get("skipped", False):
                continue
            # unknown dict shape -> skip safely
            continue

        # Normal record format: list[(word, s1, s2, diff)]
        if not isinstance(entry, list):
            continue

        for word, s1, s2, diff in entry:
            word_diffs.setdefault(word, []).append((s1, s2, diff))

    stats = []
    for word, deltas in word_diffs.items():
        s1_vals = [x[0] for x in deltas]
        s2_vals = [x[1] for x in deltas]
        diff_vals = [x[2] for x in deltas]

        mean_s1 = float(np.mean(s1_vals))
        mean_s2 = float(np.mean(s2_vals))
        mean_diff = float(np.mean(diff_vals))
        std_diff = float(np.std(diff_vals))

        stats.append((word, mean_s1, mean_s2, mean_diff, std_diff))

    stats.sort(key=lambda x: abs(x[3]), reverse=True)
    return stats[:top_n]




# -----------------------------
# Experimental code for token clustering and analysis
# -----------------------------

def build_token_cluster_summary(
    results_df,
    model_name,
    n_clusters,
    batch_size=256,
    random_state=42,
    score_column="Score",
    device="cpu",
):
    required_columns = {"Token"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"results_df is missing required columns: {missing_display}")

    tokens = results_df["Token"].dropna().astype(str).str.strip()
    unique_tokens = tokens[tokens.ne("")].drop_duplicates().tolist()

    if not unique_tokens:
        raise ValueError("results_df does not contain any non-empty tokens.")
    if n_clusters < 1 or n_clusters > len(unique_tokens):
        raise ValueError(
            f"n_clusters must be between 1 and {len(unique_tokens)} for the current results_df."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()

    def embed_batch(token_batch):
        encoded = tokenizer(
            token_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state

        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.cpu().numpy().astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=max(batch_size, n_clusters),
        n_init=10,
    )

    for start_idx in range(0, len(unique_tokens), batch_size):
        batch_tokens = unique_tokens[start_idx:start_idx + batch_size]
        batch_embeddings = embed_batch(batch_tokens)
        kmeans.partial_fit(batch_embeddings)

    token_rows = []
    distance_rows = []
    embedding_rows = []
    centers = kmeans.cluster_centers_.astype(np.float32)

    for start_idx in range(0, len(unique_tokens), batch_size):
        batch_tokens = unique_tokens[start_idx:start_idx + batch_size]
        batch_embeddings = embed_batch(batch_tokens)
        batch_labels = kmeans.predict(batch_embeddings)
        batch_centers = centers[batch_labels]
        batch_distances = np.linalg.norm(batch_embeddings - batch_centers, axis=1)

        token_rows.extend(zip(batch_tokens, batch_labels))
        distance_rows.extend(zip(batch_tokens, batch_labels, batch_distances))
        embedding_rows.extend(zip(batch_tokens, batch_embeddings.astype(np.float16)))

    token_embeddings_df = pd.DataFrame(token_rows, columns=["Token", "cluster"] )
    token_centroid_distance_df = pd.DataFrame(
        distance_rows,
        columns=["Token", "cluster", "centroid_distance"],
    )
    token_embedding_values_df = pd.DataFrame(
        embedding_rows,
        columns=["Token", "embedding"],
    )
    token_embeddings_df = token_embeddings_df.merge(token_embedding_values_df, on="Token", how="left")

    top_tokens_per_cluster_df = (
        token_centroid_distance_df
        .sort_values(["cluster", "centroid_distance", "Token"])
        .groupby("cluster")["Token"]
        .apply(lambda token_series: token_series.head(5).tolist())
        .reset_index(name="top_tokens")
    )

    embedding_matrix = np.vstack(token_embeddings_df["embedding"].to_numpy()).astype(np.float32)
    pca = PCA(n_components=2, random_state=random_state)
    token_embedding_2d = pca.fit_transform(embedding_matrix)
    token_embeddings_df["emb_x"] = token_embedding_2d[:, 0]
    token_embeddings_df["emb_y"] = token_embedding_2d[:, 1]

    clustered_results_df = results_df.copy()
    clustered_results_df["Token"] = clustered_results_df["Token"].astype(str).str.strip()
    clustered_results_df = clustered_results_df.merge(
        token_embeddings_df[["Token", "cluster"]],
        on="Token",
        how="left",
    )

    group_columns = ["cluster"]
    if "Target" in clustered_results_df.columns:
        group_columns.append("Target")

    aggregation_kwargs = {
        "token_count": ("Token", "nunique"),
        "row_count": ("Token", "size"),
    }
    if score_column in clustered_results_df.columns:
        aggregation_kwargs["avg_score"] = (score_column, "mean")

    cluster_summary_df = (
        clustered_results_df
        .groupby(group_columns, dropna=False)
        .agg(**aggregation_kwargs)
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )

    cluster_summary_df = cluster_summary_df.merge(
        top_tokens_per_cluster_df,
        on="cluster",
        how="left",
    )

    centroid_2d = pca.transform(centers)
    centroid_df = pd.DataFrame(
        {
            "cluster": np.arange(n_clusters),
            "centroid": [centroid for centroid in centers],
            "centroid_x": centroid_2d[:, 0],
            "centroid_y": centroid_2d[:, 1],
        }
    )

    return token_embeddings_df, clustered_results_df, cluster_summary_df, centroid_df