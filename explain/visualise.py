from html import escape
from IPython.display import HTML, display
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .explain import compare_explainers
from .analyse import  analyze_comparison
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

def _attr_to_rgba(score, max_abs):
    if max_abs <= 0:
        return "rgba(128,128,128,0.10)"
    strength = min(abs(score) / max_abs, 1.0)
    alpha = 0.10 + 0.75 * strength
    if score >= 0:
        return f"rgba(30, 136, 229, {alpha:.3f})"
    return f"rgba(229, 57, 53, {alpha:.3f})"

def highlight_context_tokens(explainer, sentence, target, word_agg="max", normalize=True, show=True):
    """
    Renders sentence with interactive token highlights.
    - Blue  = token supports target prediction
    - Red   = token opposes target prediction
    - Gold  = the [MASK] position, labelled with the target word
    Hovering any token shows its attribution score.
    """
    out = explainer.explain(
        [sentence],
        [[target]],
        normalize=normalize,
        return_word_scores=True,
        word_agg=word_agg
    )[0][target]

    if out.get("skipped", False):
        msg = f"Skipped: {out.get('reason', 'unknown reason')}"
        if show:
            display(HTML(f"<pre>{escape(msg)}</pre>"))
        return msg

    rows = out["word_attributions"]  # [(word, score), ...]
    mask_tok = getattr(explainer.tokenizer, "mask_token", "[MASK]")
    max_abs = max((abs(s) for w, s in rows if w != mask_tok), default=0.0)

    container_id = f"tokviz_{uuid.uuid4().hex}"

    token_spans = []
    for word, score in rows:
        if word == mask_tok:
            # Gold pill showing [MASK] → target
            span = (
                f"<span class='tok' data-score='(masked position)' "
                f"style='background:rgba(255,193,7,0.85); color:#000; font-weight:bold; "
                f"padding:2px 8px; margin:1px; border-radius:4px; cursor:default; "
                f"outline: 2px solid rgba(200,150,0,0.8);'>"
                f"[{escape(target)}]</span>"
            )
        else:
            color = _attr_to_rgba(score, max_abs)
            span = (
                f"<span class='tok' data-score='{score:.6f}' "
                f"style='background:{color}; padding:2px 4px; margin:1px; "
                f"border-radius:4px; cursor:default;'>"
                f"{escape(word)}</span>"
            )
        token_spans.append(span)

    html = f"""
    <div id="{container_id}">
      <div style='margin-bottom:6px;'>
        <b>Target:</b> <code>{escape(str(target))}</code>
      </div>
      <div style='margin:6px 0 10px 0; font-size:13px; display:flex; gap:10px; align-items:center;'>
        <span style='background:rgba(30,136,229,0.35); padding:2px 8px; border-radius:4px;'>&#9646; predicts</span>
        <span style='background:rgba(229,57,53,0.35);  padding:2px 8px; border-radius:4px;'>&#9646; opposes</span>
        <span style='background:rgba(255,193,7,0.85);  padding:2px 8px; border-radius:4px; font-weight:bold;'>[target] mask position</span>
      </div>
      <div style='line-height:2.4; font-size:15px;'>
        {' '.join(token_spans)}
      </div>
      <div class='tok-tooltip'
           style='display:none; position:fixed; z-index:9999; pointer-events:none;
                  background:#111; color:#fff; padding:5px 10px;
                  border-radius:6px; font-size:12px; font-family:monospace;'>
      </div>
    </div>
    <script>
    (function() {{
      const root = document.getElementById("{container_id}");
      if (!root) return;
      const tip  = root.querySelector(".tok-tooltip");
      const toks = root.querySelectorAll(".tok");

      toks.forEach(el => {{
        el.addEventListener("mouseenter", () => {{
          const raw = el.dataset.score;
          const parsed = parseFloat(raw);
          tip.textContent = isNaN(parsed)
            ? raw
            : "score = " + parsed.toFixed(4) + (parsed > 0 ? "  ▲ predicts" : "  ▼ opposes");
          tip.style.display = "block";
        }});
        el.addEventListener("mousemove", (e) => {{
          tip.style.left = (e.clientX + 14) + "px";
          tip.style.top  = (e.clientY + 14) + "px";
        }});
        el.addEventListener("mouseleave", () => {{
          tip.style.display = "none";
        }});
      }});
    }})();
    </script>
    """

    if show:
        display(HTML(html))
    return html

def highlight_context_tokens_multi_target(explainer, sentence, targets, word_agg="max", normalize=True):
    """
    Renders one highlighted sentence per target.
    Returns dict[target] -> html string.
    """
    rendered = {}
    for t in targets:
        rendered[t] = highlight_context_tokens(
            explainer, sentence, t, word_agg=word_agg, normalize=normalize, show=True
        )
    return rendered

def _iter_comparison_rows(comparison, target):
    """
    Yields (sent_idx, rows) where rows is list[(word, s1, s2, diff)].
    Safely skips malformed / skipped entries.
    """
    for sent_idx, sent_comp in enumerate(comparison):
        if not isinstance(sent_comp, dict) or target not in sent_comp:
            continue

        entry = sent_comp[target]

        # Skip entries like {"skipped": True, ...}
        if isinstance(entry, dict):
            if entry.get("skipped", False):
                continue
            continue

        # Normal entries are list[(word, s1, s2, diff)]
        if not isinstance(entry, list):
            continue

        rows = []
        for item in entry:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                word, s1, s2, diff = item
                rows.append((word, float(s1), float(s2), float(diff)))

        if rows:
            yield sent_idx, rows


def plot_model_comparison_bar(comparison, target, top_n=15):
    stats = analyze_comparison(comparison, target, top_n=top_n)
    if not stats:
        print(f"No valid comparison rows found for target='{target}'.")
        return

    words = [s[0] for s in stats]
    m1_scores = [s[1] for s in stats]
    m2_scores = [s[2] for s in stats]

    x = np.arange(len(words))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(x - width / 2, m1_scores, width, label="Model 1 (1760-1900)", alpha=0.8)
    ax.barh(x + width / 2, m2_scores, width, label="Model 2 (1760-1850)", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(words)
    ax.set_xlabel("Attribution Score", fontsize=12)
    ax.set_title(f"Model Comparison: Top {top_n} Predictors for '{target}'", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_scatter_model_comparison(comparison, target, top_n=25):
    sent_data = []
    for _, rows in _iter_comparison_rows(comparison, target):
        for word, s1, s2, _ in rows:
            sent_data.append((word, s1, s2))

    if not sent_data:
        print(f"No valid comparison rows found for target='{target}'.")
        return

    word_agg = {}
    for word, s1, s2 in sent_data:
        word_agg.setdefault(word, {"s1": [], "s2": []})
        word_agg[word]["s1"].append(s1)
        word_agg[word]["s2"].append(s2)

    word_means = []
    for word, vals in word_agg.items():
        m1 = float(np.mean(vals["s1"]))
        m2 = float(np.mean(vals["s2"]))
        word_means.append((word, m1, m2, abs(m2 - m1)))

    word_means.sort(key=lambda x: x[3], reverse=True)
    word_means = word_means[:top_n]

    words = [w[0] for w in word_means]
    m1_vals = [w[1] for w in word_means]
    m2_vals = [w[2] for w in word_means]
    diffs = [w[3] for w in word_means]

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(m1_vals, m2_vals, s=200, c=diffs, cmap="YlOrRd", alpha=0.6, edgecolors="black", linewidth=1)

    for i, word in enumerate(words):
        ax.annotate(word, (m1_vals[i], m2_vals[i]), fontsize=9, ha="center", va="center")

    lim_min = min(min(m1_vals), min(m2_vals)) * 0.9
    lim_max = max(max(m1_vals), max(m2_vals)) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.3, linewidth=2, label="Equal scores")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Model 1 Attribution Score (1760-1900)", fontsize=12)
    ax.set_ylabel("Model 2 Attribution Score (1760-1850)", fontsize=12)
    ax.set_title(f"Model Attribution Comparison for '{target}'", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.colorbar(scatter, ax=ax, label="Absolute Difference")
    ax.legend()
    plt.tight_layout()
    plt.show()



def export_comparison_csv(comparison, target, output_file="model_comparison.csv"):
    rows_out = []
    for sent_idx, rows in _iter_comparison_rows(comparison, target):
        for word, s1, s2, diff in rows:
            rows_out.append({
                "sentence_idx": sent_idx,
                "word": word,
                "model1_score": float(s1),
                "model2_score": float(s2),
                "difference": float(diff),
            })

    comp_df = pd.DataFrame(rows_out)
    comp_df.to_csv(output_file, index=False)
    print(f"Comparison exported to {output_file} ({len(comp_df)} rows)")
    return comp_df



def _safe_sentence_rows(comparison, sent_idx, target):
    """Return normalized rows: [(word, old_score, new_score, diff), ...] or None."""
    if sent_idx < 0 or sent_idx >= len(comparison):
        return None
    sent_comp = comparison[sent_idx]
    if not isinstance(sent_comp, dict) or target not in sent_comp:
        return None

    entry = sent_comp[target]
    if isinstance(entry, dict):   # skipped/malformed record
        return None
    if not isinstance(entry, list):
        return None

    rows = []
    for item in entry:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            w, old_s, new_s, d = item
            rows.append((str(w), float(old_s), float(new_s), float(d)))
    return rows if rows else None

def render_top_shift_sentences(
    texts,
    comparison,
    target,
    top_k=5,
    score_mode="mean_abs",   # "mean_abs" or "max_abs"
    show=True
):
    """
    Render top sentences where model change is largest for a given target.
    comparison rows are expected as: (word, old_score, new_score, diff=old-new).
    Blue = toward old model (diff > 0), Red = toward new model (diff < 0).
    """
    ranked = []
    for i, sent in enumerate(texts):
        rows = _safe_sentence_rows(comparison, i, target)
        if not rows:
            continue
        diffs = np.array([r[3] for r in rows], dtype=float)
        shift = float(np.mean(np.abs(diffs))) if score_mode == "mean_abs" else float(np.max(np.abs(diffs)))
        ranked.append((i, sent, rows, shift))

    ranked.sort(key=lambda x: x[3], reverse=True)
    ranked = ranked[:top_k]

    if not ranked:
        msg = f"No valid rows found for target='{target}'."
        if show:
            print(msg)
        return []

    rendered = []
    for rank, (sent_idx, sent, rows, shift) in enumerate(ranked, start=1):
        container_id = f"shiftviz_{uuid.uuid4().hex}"

        # normalize intensity by |diff|
        max_abs = max(abs(d) for _, _, _, d in rows) if rows else 0.0
        max_abs = max(max_abs, 1e-12)

        token_spans = []
        for w, old_s, new_s, d in rows:
            if w == "[MASK]":
                span = (
                    f"<span class='tok' data-tip='[MASK] position | target={escape(str(target))}' "
                    f"style='background:rgba(255,193,7,0.90); color:#111; font-weight:bold; "
                    f"padding:2px 8px; margin:1px; border-radius:4px; outline:2px solid rgba(200,150,0,0.9);'>"
                    f"[{escape(str(target))}]</span>"
                )
            else:
                strength = min(abs(d) / max_abs, 1.0)
                alpha = 0.12 + 0.78 * strength
                # diff = old - new; positive => old stronger (blue), negative => new stronger (red)
                bg = f"rgba(30,136,229,{alpha:.3f})" if d > 0 else f"rgba(229,57,53,{alpha:.3f})"
                tip = f"{escape(w)} | old={old_s:.4f} | new={new_s:.4f} | diff(old-new)={d:.4f}"
                span = (
                    f"<span class='tok' data-tip='{tip}' "
                    f"style='background:{bg}; padding:2px 4px; margin:1px; border-radius:4px; cursor:default;'>"
                    f"{escape(w)}</span>"
                )
            token_spans.append(span)

        html = f"""
        <div id="{container_id}" style="margin:10px 0 18px 0;">
          <div style="margin-bottom:6px;">
            <b>#{rank}</b> sentence_idx=<code>{sent_idx}</code> | shift=<code>{shift:.4f}</code>
          </div>
          <div style="margin-bottom:6px; color:#444;">
            {escape(sent)}
          </div>
          <div style="margin:6px 0 10px 0; font-size:13px; display:flex; gap:8px; align-items:center;">
            <span style="background:rgba(30,136,229,0.35); padding:2px 8px; border-radius:4px;">blue: toward old</span>
            <span style="background:rgba(229,57,53,0.35); padding:2px 8px; border-radius:4px;">red: toward new</span>
            <span style="background:rgba(255,193,7,0.90); padding:2px 8px; border-radius:4px; font-weight:bold;">[target] mask</span>
          </div>
          <div style="line-height:2.3; font-size:15px;">
            {' '.join(token_spans)}
          </div>
          <div class="tok-tooltip"
               style="display:none; position:fixed; z-index:9999; pointer-events:none;
                      background:#111; color:#fff; padding:6px 10px; border-radius:6px;
                      font-size:12px; font-family:monospace; max-width:70vw; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
          </div>
        </div>
        <script>
        (function() {{
          const root = document.getElementById("{container_id}");
          if (!root) return;
          const tip = root.querySelector(".tok-tooltip");
          root.querySelectorAll(".tok").forEach(el => {{
            el.addEventListener("mouseenter", () => {{
              tip.textContent = el.dataset.tip || "";
              tip.style.display = "block";
            }});
            el.addEventListener("mousemove", (e) => {{
              tip.style.left = (e.clientX + 14) + "px";
              tip.style.top  = (e.clientY + 14) + "px";
            }});
            el.addEventListener("mouseleave", () => {{
              tip.style.display = "none";
            }});
          }});
        }})();
        </script>
        """
        rendered.append(html)
        if show:
            display(HTML(html))

    return rendered

# -----------------------------
# Experimental code for token embedding visualization
# -----------------------------


def plot_token_embeddings_interactive(
    token_embeddings_df,
    perplexity=30,
    random_state=42,
    point_size=3,
    opacity=0.75,
):
    required_cols = {"cluster", "Token", "embedding"}
    missing_cols = required_cols - set(token_embeddings_df.columns)
    if missing_cols:
        missing_display = ", ".join(sorted(missing_cols))
        raise ValueError(f"token_embeddings_df is missing required columns: {missing_display}")

    plot_df = token_embeddings_df[["Token", "cluster", "embedding"]].copy()
    plot_df["cluster"] = plot_df["cluster"].astype(str)

    embedding_matrix = np.vstack(plot_df["embedding"].to_numpy()).astype(np.float32)
    n_samples = embedding_matrix.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 tokens to compute t-SNE.")

    valid_perplexity = min(perplexity, max(2, n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=valid_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    token_embedding_2d = tsne.fit_transform(embedding_matrix)
    plot_df["tsne_x"] = token_embedding_2d[:, 0]
    plot_df["tsne_y"] = token_embedding_2d[:, 1]

    fig = px.scatter(
        plot_df,
        x="tsne_x",
        y="tsne_y",
        color="cluster",
        hover_data=["Token"],
        title="Token Embeddings (t-SNE) by Cluster",
        opacity=opacity,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(
        marker={"size": point_size, "line": {"width": 0}},
        selector={"mode": "markers"},
    )

    center_df = (
        plot_df.groupby("cluster", as_index=False)[["tsne_x", "tsne_y"]]
        .mean()
        .sort_values("cluster")
    )
    fig.add_trace(
        go.Scatter(
            x=center_df["tsne_x"],
            y=center_df["tsne_y"],
            mode="markers+text",
            text=center_df["cluster"],
            textposition="top center",
            marker={
                "symbol": "x",
                "size": 14,
                "line": {"width": 2},
            },
            name="cluster centers (t-SNE mean)",
            hovertemplate="cluster=%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend_title_text="Cluster",
        template="plotly_white",
    )
    return fig