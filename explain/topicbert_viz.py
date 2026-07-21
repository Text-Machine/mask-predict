from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from plotly.colors import hex_to_rgb
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


def embed_sentences(
    sentences: List[str],
    checkpoint: str,
    batch_size: int = 32,
    normalize: bool = True,
    device: str | None = None,
) -> torch.Tensor:
    """Embed a list of sentences using a Hugging Face checkpoint."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for index in range(0, len(sentences), batch_size):
            batch = sentences[index : index + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)

            pooled = (token_embeddings * attention_mask).sum(dim=1)
            pooled /= attention_mask.sum(dim=1).clamp(min=1)

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)

            embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)


def draw_scatterplot_regions(
    fig,
    data,
    region_col,
    x_col="tsne_x",
    y_col="tsne_y",
    label_col=None,
    min_points=3,
    fill_alpha=0.12,
    line_width=2,
):
    """Draw filled region outlines for categorical groups on a 2D scatterplot."""

    def to_rgba(color_hex, alpha):
        red, green, blue = hex_to_rgb(color_hex)
        return f"rgba({red}, {green}, {blue}, {alpha})"

    include_label_col = [label_col] if label_col and label_col not in [region_col] else []
    region_data = data[[x_col, y_col, region_col] + include_label_col].dropna(subset=[x_col, y_col, region_col]).copy()

    region_data = region_data[~region_data[region_col].isin([-1, "-1", "noise", "Noise", None])]
    if region_data.empty:
        return fig

    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
    groups = sorted(region_data.groupby(region_col), key=lambda item: len(item[1]), reverse=True)

    for index, (region_name, region_points) in enumerate(groups):
        if len(region_points) < min_points:
            continue

        coordinates = region_points[[x_col, y_col]].to_numpy()
        color = palette[index % len(palette)]
        fill_color = to_rgba(color, fill_alpha)
        line_color = to_rgba(color, 0.95)

        try:
            hull = ConvexHull(coordinates)
            polygon = coordinates[hull.vertices]
        except Exception:
            x_min, y_min = coordinates.min(axis=0)
            x_max, y_max = coordinates.max(axis=0)
            x_pad = max((x_max - x_min) * 0.08, 0.15)
            y_pad = max((y_max - y_min) * 0.08, 0.15)
            polygon = np.array(
                [
                    [x_min - x_pad, y_min - y_pad],
                    [x_max + x_pad, y_min - y_pad],
                    [x_max + x_pad, y_max + y_pad],
                    [x_min - x_pad, y_max + y_pad],
                ]
            )

        path = "M " + " L ".join(f"{x},{y}" for x, y in polygon) + " Z"
        fig.add_shape(
            type="path",
            path=path,
            xref="x",
            yref="y",
            fillcolor=fill_color,
            line=dict(color=line_color, width=line_width),
            layer="below",
        )

        centroid = region_points[[x_col, y_col]].mean()
        region_label = region_name
        if label_col and label_col in region_points.columns:
            region_label = region_points[label_col].iloc[0]

        fig.add_annotation(
            x=centroid[x_col],
            y=centroid[y_col],
            text=f"Region {region_label} ({len(region_points)})",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=line_color,
            borderwidth=1,
            font=dict(size=11, color="black"),
        )

    return fig


def cluster_scatterplot_points(
    data,
    x_col="tsne_x",
    y_col="tsne_y",
    method="kmeans",
    n_clusters=None,
    cluster_range=range(3, 9),
    min_points=3,
    random_state=42,
):
    """Assign cluster labels to 2D scatterplot points with a distinct-cluster bias."""
    cluster_data = data[[x_col, y_col]].dropna().copy()
    if cluster_data.empty:
        cluster_data["cluster_label"] = []
        return cluster_data

    points = cluster_data[[x_col, y_col]].to_numpy()
    scaled_points = StandardScaler().fit_transform(points) if len(cluster_data) > 1 else points

    def fit_labels(cluster_count):
        if method == "kmeans":
            model = KMeans(n_clusters=cluster_count, n_init="auto", random_state=random_state)
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
        else:
            raise ValueError("method must be 'kmeans' or 'agglomerative'")
        return model.fit_predict(scaled_points)

    if n_clusters is None:
        candidates = [cluster_count for cluster_count in cluster_range if 1 < cluster_count < len(cluster_data)]
        if not candidates:
            n_clusters = 2 if len(cluster_data) > 1 else 1
        else:
            best_score = -np.inf
            best_candidate = candidates[0]
            for cluster_count in candidates:
                labels = fit_labels(cluster_count)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(scaled_points, labels)
                if score > best_score:
                    best_score = score
                    best_candidate = cluster_count
            n_clusters = best_candidate

    cluster_data["cluster_label"] = fit_labels(n_clusters)

    if min_points > 1:
        counts = cluster_data["cluster_label"].value_counts()
        small_clusters = counts[counts < min_points].index
        cluster_data.loc[cluster_data["cluster_label"].isin(small_clusters), "cluster_label"] = -1

    return cluster_data


def draw_clustered_scatterplot_regions(
    fig,
    data,
    x_col="tsne_x",
    y_col="tsne_y",
    method="kmeans",
    n_clusters=None,
    cluster_range=range(3, 9),
    min_points=3,
    label_col=None,
    fill_alpha=0.12,
    line_width=2,
):
    """Cluster 2D points first, then draw the resulting regions on the scatterplot."""
    clustered_data = cluster_scatterplot_points(
        data,
        x_col=x_col,
        y_col=y_col,
        method=method,
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        min_points=min_points,
    )

    cluster_summary = (
        clustered_data["cluster_label"]
        .value_counts()
        .rename_axis("cluster_label")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print(f"Clustering method: {method}")
    print(cluster_summary.to_string(index=False))

    figure_with_regions = draw_scatterplot_regions(
        fig,
        clustered_data,
        region_col="cluster_label",
        x_col=x_col,
        y_col=y_col,
        label_col=label_col,
        fill_alpha=fill_alpha,
        line_width=line_width,
    )
    return figure_with_regions, clustered_data, cluster_summary


def _topicbert_get_encoder(model):
    if hasattr(model, "base_model") and model.base_model is not None:
        return model.base_model
    if hasattr(model, "bert"):
        return model.bert
    if hasattr(model, "roberta"):
        return model.roberta
    return model


def _topicbert_mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    masked = hidden_states * mask
    denominator = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denominator


def _topicbert_embed_texts(
    texts,
    model,
    tokenizer,
    batch_size=8,
    max_length=96,
    device="cpu",
):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    encoder = _topicbert_get_encoder(model).to(device)
    encoder.eval()
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = encoder(**encoded, return_dict=True)
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = _topicbert_mean_pool(hidden_states, encoded["attention_mask"])
            embeddings.append(pooled.detach().cpu().numpy())

    return np.vstack(embeddings)


def _topicbert_pick_n_clusters(embeddings, method="kmeans", cluster_range=range(3, 9), random_state=42):
    if len(embeddings) < 3:
        return 1

    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    candidates = [candidate for candidate in cluster_range if 1 < candidate < len(embeddings)]
    if not candidates:
        return min(2, len(embeddings))

    best_candidate = candidates[0]
    best_score = -np.inf

    for candidate in candidates:
        if method == "kmeans":
            labels = KMeans(n_clusters=candidate, n_init="auto", random_state=random_state).fit_predict(scaled_embeddings)
        elif method == "agglomerative":
            labels = AgglomerativeClustering(n_clusters=candidate, linkage="ward").fit_predict(scaled_embeddings)
        else:
            raise ValueError("method must be 'kmeans' or 'agglomerative'")

        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(scaled_embeddings, labels)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def _topicbert_cluster_embeddings(
    embeddings,
    method="kmeans",
    n_clusters=None,
    cluster_range=range(3, 9),
    random_state=42,
):
    if len(embeddings) == 0:
        return np.array([], dtype=int), 1

    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    if n_clusters is None:
        n_clusters = _topicbert_pick_n_clusters(
            embeddings,
            method=method,
            cluster_range=cluster_range,
            random_state=random_state,
        )

    if n_clusters <= 1:
        return np.zeros(len(embeddings), dtype=int), n_clusters

    if method == "kmeans":
        labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state).fit_predict(scaled_embeddings)
    elif method == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(scaled_embeddings)
    else:
        raise ValueError("method must be 'kmeans' or 'agglomerative'")

    return labels, n_clusters


def _topicbert_document_topic_distributions(embeddings, labels):
    """Build soft per-document topic probabilities from distances to topic centroids."""
    if len(embeddings) == 0:
        return pd.DataFrame()

    topic_ids = np.sort(np.unique(labels.astype(int)))
    scaled_embeddings = StandardScaler().fit_transform(embeddings)

    if len(topic_ids) == 1:
        only_topic = int(topic_ids[0])
        return pd.DataFrame(
            {
                "dominant_topic_label": np.full(len(embeddings), only_topic, dtype=int),
                f"topic_prob_{only_topic}": np.ones(len(embeddings), dtype=float),
            }
        )

    centroids = np.vstack(
        [scaled_embeddings[labels == topic_id].mean(axis=0) for topic_id in topic_ids]
    )
    squared_distances = ((scaled_embeddings[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    logits = -squared_distances
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    probability_columns = {
        f"topic_prob_{int(topic_id)}": probabilities[:, column_index]
        for column_index, topic_id in enumerate(topic_ids)
    }
    dominant_topic_ids = topic_ids[np.argmax(probabilities, axis=1)].astype(int)

    topic_distributions = pd.DataFrame(probability_columns)
    topic_distributions.insert(0, "dominant_topic_label", dominant_topic_ids)
    return topic_distributions


def _topicbert_topic_keywords(text_frame, text_col, topic_col="topic_label", top_n=8):
    grouped_texts = text_frame.groupby(topic_col)[text_col].apply(lambda series: " ".join(series.astype(str))).sort_index()
    if grouped_texts.empty:
        return pd.DataFrame(columns=[topic_col, "topic_size", "topic_name", "top_terms", "example_sentence"])

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=3000)
    topic_matrix = vectorizer.fit_transform(grouped_texts.tolist())
    terms = np.asarray(vectorizer.get_feature_names_out())

    rows = []
    for row_index, topic_id in enumerate(grouped_texts.index):
        weights = topic_matrix[row_index].toarray().ravel()
        if weights.sum() == 0:
            top_terms = []
        else:
            top_indices = np.argsort(weights)[::-1][:top_n]
            top_terms = [terms[index] for index in top_indices if weights[index] > 0]

        topic_rows = text_frame[text_frame[topic_col] == topic_id]
        example_sentence = topic_rows[text_col].iloc[0] if not topic_rows.empty else ""
        topic_name = ", ".join(top_terms[:3]) if top_terms else f"Topic {topic_id}"

        rows.append(
            {
                topic_col: topic_id,
                "topic_size": len(topic_rows),
                "topic_name": topic_name,
                "top_terms": ", ".join(top_terms),
                "example_sentence": example_sentence,
            }
        )

    topic_summary = pd.DataFrame(rows).sort_values(["topic_size", topic_col], ascending=[False, True]).reset_index(drop=True)
    return topic_summary


def plot_topicbert_topics(
    model,
    dataframe,
    text_column,
    tokenizer,
    max_rows=200,
    batch_size=8,
    max_length=96,
    embed_device="cpu",
    cluster_method="kmeans",
    n_clusters=None,
    cluster_range=range(3, 9),
    random_state=42,
):
    """Build a TopicBERT-style topic map from BERT sentence embeddings.

    Returns:
        topic_frame: Document-level dataframe with hard labels and 2D coordinates.
        topic_summary: Topic-level summary with size and keyword descriptions.
        topic_distributions: Per-document soft topic distributions.
        fig: Plotly figure of the 2D topic map.
    """
    if tokenizer is None:
        raise ValueError("A tokenizer is required")

    if text_column not in dataframe.columns:
        raise KeyError(f"Column '{text_column}' not found in dataframe")

    topic_frame = dataframe[[text_column]].dropna().copy()
    topic_frame[text_column] = topic_frame[text_column].astype(str).str.strip()
    topic_frame = topic_frame[topic_frame[text_column] != ""]
    if max_rows is not None and max_rows > 0:
        topic_frame = topic_frame.head(max_rows).copy()

    if topic_frame.empty:
        raise ValueError("No non-empty text rows available for TopicBERT")

    texts = topic_frame[text_column].tolist()
    embeddings = _topicbert_embed_texts(
        texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=embed_device,
    )

    if len(embeddings) > 2:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(StandardScaler().fit_transform(embeddings))
    else:
        coords = np.zeros((len(embeddings), 2), dtype=float)

    labels, chosen_clusters = _topicbert_cluster_embeddings(
        embeddings,
        method=cluster_method,
        n_clusters=n_clusters,
        cluster_range=cluster_range,
        random_state=random_state,
    )

    topic_frame = topic_frame.reset_index(drop=True)
    topic_frame["topic_label"] = labels
    topic_frame["tsne_x"] = coords[:, 0]
    topic_frame["tsne_y"] = coords[:, 1]

    topic_summary = _topicbert_topic_keywords(topic_frame, text_column, topic_col="topic_label", top_n=8)
    topic_summary["topic_label"] = topic_summary["topic_label"].astype(int)

    topic_distributions = _topicbert_document_topic_distributions(embeddings, labels)
    topic_distributions.insert(0, text_column, topic_frame[text_column].values)
    topic_distributions.insert(0, "document_index", np.arange(len(topic_frame), dtype=int))
    topic_distributions["assigned_topic_label"] = topic_frame["topic_label"].astype(int).values

    name_map = dict(zip(topic_summary["topic_label"], topic_summary["topic_name"]))
    topic_distributions["dominant_topic_name"] = topic_distributions["dominant_topic_label"].map(name_map)
    topic_distributions["assigned_topic_name"] = topic_distributions["assigned_topic_label"].map(name_map)

    topic_frame = topic_frame.merge(topic_summary[["topic_label", "topic_name", "top_terms"]], on="topic_label", how="left")

    print(f"TopicBERT clustering method: {cluster_method}")
    print(f"Chosen clusters: {chosen_clusters}")
    print(topic_summary[["topic_label", "topic_size", "topic_name"]].to_string(index=False))

    fig = px.scatter(
        topic_frame,
        x="tsne_x",
        y="tsne_y",
        color="topic_name",
        hover_data={
            text_column: True,
            "topic_label": True,
            "top_terms": True,
            "tsne_x": False,
            "tsne_y": False,
        },
        title="TopicBERT: sentence topics from BERT embeddings",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(marker=dict(size=0.5, opacity=0.85, line=dict(width=0.5, color="white")))
    fig.update_layout(legend_title_text="Topic")

    return topic_frame, topic_summary, topic_distributions, fig


__all__ = [
    "embed_sentences",
    "draw_scatterplot_regions",
    "cluster_scatterplot_points",
    "draw_clustered_scatterplot_regions",
    "plot_topicbert_topics",
]
