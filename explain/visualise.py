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
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import squarify

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


def _attr_to_mpl_rgba(score, max_abs):
    """Same colour logic as `_attr_to_rgba`, but as a matplotlib-ready (r,g,b,a) tuple."""
    if max_abs <= 0:
        return (0.5, 0.5, 0.5, 0.10)
    strength = min(abs(score) / max_abs, 1.0)
    alpha = 0.15 + 0.65 * strength
    if score >= 0:
        return (30 / 255, 136 / 255, 229 / 255, alpha)
    return (229 / 255, 57 / 255, 53 / 255, alpha)


def _measure_text(measure_fig, renderer, text, fontsize, weight="normal"):
    """Width/height (in inches) that `text` would occupy at `fontsize` on `measure_fig`."""
    t = measure_fig.text(0, 0, text, fontsize=fontsize, fontweight=weight)
    bbox = t.get_window_extent(renderer=renderer)
    t.remove()
    return bbox.width / measure_fig.dpi, bbox.height / measure_fig.dpi


def _merge_diff_rows(rows1, rows2, mask_tok):
    """
    Merge the word_attributions of two targets on the *same* sentence into a
    single list[(word, diff_score)] with diff = score(target1) - score(target2),
    collapsing the (possibly multi-token) [MASK] block into one sentinel
    entry `(mask_tok, None)` at the position it occurred.

    Raises ValueError if the non-mask context tokens don't line up 1:1 (this
    would only happen if the two target words somehow led to different
    tokenization of the surrounding context, which shouldn't normally occur).
    """
    ctx1 = [(w, s) for w, s in rows1 if w != mask_tok]
    ctx2 = [(w, s) for w, s in rows2 if w != mask_tok]
    words1 = [w for w, _ in ctx1]
    words2 = [w for w, _ in ctx2]
    if words1 != words2:
        raise ValueError(
            "context tokens differ between the two target words, cannot compute "
            f"a direct difference: {words1} vs {words2}"
        )

    mask_idx = next((i for i, (w, _) in enumerate(rows1) if w == mask_tok), None)
    insert_at = (
        sum(1 for w, _ in rows1[:mask_idx] if w != mask_tok) if mask_idx is not None else len(ctx1)
    )

    merged = [(w, s1 - s2) for (w, s1), (_, s2) in zip(ctx1, ctx2)]
    merged.insert(insert_at, (mask_tok, None))
    return merged


def _wrap_and_layout_sentences(
    measure_fig, renderer, results, fontsize, x_left, x_right, top_y, bottom_y,
    mask_tok, color_scale, global_max_abs,
    line_spacing=1.55, header_gap=1.05, row_gap=0.95, sentence_gap=1.5,
    token_pad_x=0.045, token_gap_x=0.035,
):
    """
    Lay out every (label, target, rows, ..., mode) entry as wrapped rows of
    token boxes, top to bottom, starting at `top_y`. Returns (ops, end_y)
    where `ops` is a list of draw instructions and `end_y` is the
    y-coordinate the cursor ended on (compare against `bottom_y` to check it
    fits the page).

    `global_max_abs` is a dict {"single": float, "diff": float} — "single"
    and "diff" mode rows are kept on separate colour scales since a raw
    attribution score and a difference-of-two-attributions aren't the same
    quantity.
    """
    line_h = (fontsize / 72.0) * line_spacing
    y = top_y
    ops = []

    for label, target, rows, skipped, reason, local_max, mode in results:
        y -= line_h * header_gap
        ops.append({
            "type": "text", "x": x_left, "y": y, "s": label,
            "fontsize": fontsize + 1, "weight": "bold", "ha": "left", "va": "center",
        })

        if skipped:
            y -= line_h * row_gap
            ops.append({
                "type": "text", "x": x_left, "y": y, "s": f"Skipped: {reason}",
                "fontsize": max(fontsize - 1, 6), "style": "italic",
                "color": (0.55, 0.1, 0.1, 1.0), "ha": "left", "va": "center",
            })
            y -= line_h * sentence_gap
            continue

        max_abs = global_max_abs[mode] if color_scale == "global" else local_max
        x = x_left
        y -= line_h * row_gap
        for word, score in rows:
            is_mask = word == mask_tok
            text = f"[{target}]" if is_mask else word
            weight = "bold" if is_mask else "normal"
            w_in, _ = _measure_text(measure_fig, renderer, text, fontsize, weight)
            box_w = w_in + 2 * token_pad_x
            box_h = line_h * 0.9

            if x + box_w > x_right and x > x_left:
                x = x_left
                y -= line_h

            color = (1.0, 0.757, 0.027, 0.9) if is_mask else _attr_to_mpl_rgba(score, max_abs)
            ops.append({
                "type": "rect", "x": x, "y": y - box_h / 2, "w": box_w, "h": box_h,
                "color": color, "edge": (0.6, 0.45, 0.0, 0.9) if is_mask else "none",
            })
            ops.append({
                "type": "text", "x": x + box_w / 2, "y": y, "s": text, "fontsize": fontsize,
                "weight": weight, "color": "black" if is_mask else (0.05, 0.05, 0.05, 1.0),
                "ha": "center", "va": "center",
            })
            x += box_w + token_gap_x

        y -= line_h * sentence_gap

    return ops, y


def plot_highlighted_sentences(
    explainer,
    items,
    word_agg="max",
    normalize=True,
    page_size="A4",
    orientation="portrait",
    fontsize=11,
    min_fontsize=6,
    margin_in=0.75,
    color_scale="global",
    title=None,
    save_path=None,
    dpi=200,
    show=True,
):
    """
    Static, print-friendly version of `highlight_context_tokens`: renders
    several (sentence, target) attribution highlights stacked vertically as
    wrapped rows of coloured token boxes on a single matplotlib page.

    Two modes per item:
    - single target: `target` is a word/string. Blue = token supports target
      prediction, Red = opposes, Gold = [MASK] position (labelled with the
      target word) — same convention as the interactive HTML version.
    - diff mode: `target` is a `[word1, word2]` pair. Each context token is
      coloured by diff = attribution(word1) − attribution(word2): Blue =
      token favours word1 over word2, Red = favours word2 over word1, Gold =
      [MASK] position (labelled "word1 → word2"). Useful for e.g. comparing
      how context supports "machine" vs "engine" in the same sentence.

    Parameters
    ----------
    explainer : object with an `.explain()` method (and `.tokenizer`), as
        used by `highlight_context_tokens`.
    items : list of (sentence, target) or (sentence, target, label) tuples,
        where `target` is either a word/string (single-target mode) or a
        `[word1, word2]` list/tuple (diff mode). `label` overrides the
        auto-generated header.
    word_agg, normalize : passed straight through to `explainer.explain`.
    page_size : "A4" or "LETTER".
    orientation : "portrait" or "landscape".
    fontsize : starting token font size in points; auto-shrunk (down to
        `min_fontsize`) so everything fits on one page.
    color_scale : "global" (default, one intensity scale shared across all
        sentences so they're comparable) or "per_sentence" (each sentence
        normalized to its own max attribution, like the interactive version).
    title : optional page title.
    save_path : optional path to save the figure (png/pdf/svg...).
    show : if True, calls plt.show().

    Returns
    -------
    (fig, ax) : the matplotlib Figure/Axes for the rendered page.
    """
    page_sizes_in = {"A4": (8.27, 11.69), "LETTER": (8.5, 11.0)}
    key = page_size.upper()
    if key not in page_sizes_in:
        raise ValueError(f"Unsupported page_size {page_size!r}; choose from {list(page_sizes_in)}")
    w_in, h_in = page_sizes_in[key]
    if orientation.lower() == "landscape":
        w_in, h_in = h_in, w_in
    elif orientation.lower() != "portrait":
        raise ValueError("orientation must be 'portrait' or 'landscape'")

    norm_items = []
    for it in items:
        if len(it) == 2:
            sent, tgt, label = it[0], it[1], None
        elif len(it) == 3:
            sent, tgt, label = it
        else:
            raise ValueError("each item must be a (sentence, target) or (sentence, target, label) tuple")
        norm_items.append((sent, tgt, label))
    if not norm_items:
        raise ValueError("items must contain at least one (sentence, target) pair")

    mask_tok = getattr(explainer.tokenizer, "mask_token", "[MASK]")

    # Pass 1: run the explainer once per pair, collect rows + a global score scale.
    # "single" and "diff" items are kept on separate colour scales (see
    # `_wrap_and_layout_sentences`), since they're not the same quantity.
    results = []
    all_scores = {"single": [], "diff": []}
    for idx, (sent, tgt, label) in enumerate(norm_items, start=1):
        is_diff = isinstance(tgt, (list, tuple))

        if is_diff:
            if len(tgt) != 2:
                raise ValueError(
                    f"diff-mode target must be a [word1, word2] pair, got {tgt!r} "
                    f"(item {idx})"
                )
            w1, w2 = tgt
            mode = "diff"
            disp_label = label or f"{idx}. target: \"{w1}\" → \"{w2}\""
            out_pair = explainer.explain(
                [sent], [[w1, w2]], normalize=normalize, return_word_scores=True, word_agg=word_agg
            )[0]
            out1, out2 = out_pair[w1], out_pair[w2]
            if out1.get("skipped", False) or out2.get("skipped", False):
                reason = out1.get("reason") if out1.get("skipped", False) else out2.get("reason")
                results.append((disp_label, f"{w1} → {w2}", [], True,
                                 reason or "unknown reason", 0.0, mode))
                continue
            try:
                rows = _merge_diff_rows(out1["word_attributions"], out2["word_attributions"], mask_tok)
            except ValueError as e:
                results.append((disp_label, f"{w1} → {w2}", [], True, str(e), 0.0, mode))
                continue
            display_target = f"{w1} → {w2}"
        else:
            mode = "single"
            disp_label = label or f"{idx}. target: \"{tgt}\""
            out = explainer.explain(
                [sent], [[tgt]], normalize=normalize, return_word_scores=True, word_agg=word_agg
            )[0][tgt]
            if out.get("skipped", False):
                results.append((disp_label, tgt, [], True, out.get("reason", "unknown reason"), 0.0, mode))
                continue
            rows = out["word_attributions"]
            display_target = tgt

        local_max = max((abs(s) for w, s in rows if w != mask_tok), default=0.0)
        all_scores[mode].extend(abs(s) for w, s in rows if w != mask_tok)
        results.append((disp_label, display_target, rows, False, None, local_max, mode))

    global_max_abs = {m: max(scores, default=0.0) for m, scores in all_scores.items()}

    # Reserve fixed space for an optional title (top) and legend (bottom); the
    # remaining band is what gets auto-fit by shrinking the font size.
    x_left, x_right = margin_in, w_in - margin_in
    top_y = h_in - margin_in - (0.45 if title else 0.0)
    bottom_y = margin_in + 0.35  # room for the legend strip

    measure_fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    measure_fig.canvas.draw()
    renderer = measure_fig.canvas.get_renderer()

    cur_fontsize = float(fontsize)
    ops, end_y = [], top_y
    for _ in range(6):
        ops, end_y = _wrap_and_layout_sentences(
            measure_fig, renderer, results, cur_fontsize,
            x_left, x_right, top_y, bottom_y, mask_tok, color_scale, global_max_abs,
        )
        if end_y >= bottom_y or cur_fontsize <= min_fontsize:
            break
        used, available = top_y - end_y, top_y - bottom_y
        cur_fontsize = max(min_fontsize, cur_fontsize * (available / used) * 0.97)
    plt.close(measure_fig)

    if end_y < bottom_y:
        print(
            f"[plot_highlighted_sentences] Warning: content overflows the {page_size} page "
            f"even at fontsize={cur_fontsize:.1f}pt (min_fontsize={min_fontsize}). "
            "Consider fewer sentences, a larger page_size/orientation, or a lower min_fontsize."
        )

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, w_in)
    ax.set_ylim(0, h_in)
    ax.axis("off")

    if title:
        ax.text(w_in / 2, h_in - margin_in / 2, title, fontsize=fontsize + 4,
                 fontweight="bold", ha="center", va="center")

    for op in ops:
        if op["type"] == "rect":
            edge = op.get("edge", "none")
            ax.add_patch(Rectangle(
                (op["x"], op["y"]), op["w"], op["h"],
                facecolor=op["color"], edgecolor=edge, linewidth=1.0 if edge != "none" else 0,
            ))
        else:
            ax.text(
                op["x"], op["y"], op["s"], fontsize=op["fontsize"],
                fontweight=op.get("weight", "normal"), style=op.get("style", "normal"),
                color=op.get("color", "black"), ha=op.get("ha", "left"), va=op.get("va", "center"),
            )

    # Legend strip along the bottom margin, tailored to which mode(s) are on the page.
    legend_y = margin_in - 0.05
    has_single = any(r[6] == "single" for r in results)
    has_diff = any(r[6] == "diff" for r in results)
    legend_items = []
    if has_single:
        legend_items += [
            ((30 / 255, 136 / 255, 229 / 255, 0.55), "predicts"),
            ((229 / 255, 57 / 255, 53 / 255, 0.55), "opposes"),
        ]
    if has_diff:
        legend_items += [
            ((30 / 255, 136 / 255, 229 / 255, 0.55), "favours word₁"),
            ((229 / 255, 57 / 255, 53 / 255, 0.55), "favours word₂"),
        ]
    legend_items.append(((1.0, 0.757, 0.027, 0.9), "[target] mask position"))
    lx = x_left
    for color, text in legend_items:
        box_w, box_h = 0.18, 0.14
        ax.add_patch(Rectangle((lx, legend_y), box_w, box_h, facecolor=color, edgecolor="none"))
        w_in_txt, _ = _measure_text(fig, fig.canvas.get_renderer(), text, 8)
        ax.text(lx + box_w + 0.06, legend_y + box_h / 2, text, fontsize=8, va="center", ha="left")
        lx += box_w + 0.06 + w_in_txt + 0.35

    if save_path:
        fig.savefig(save_path, dpi=dpi)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()

    return fig, ax


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

"""
Visualize the value counts of a categorical column ('semantic') as a
proportional-square (treemap-style) chart, with the value name and its
share printed on each square.

Requires: pandas, matplotlib, squarify
    pip install squarify --break-system-packages
"""



def _label_lines(name, count, pct):
    return [f"{name}", f"{count:,} ({pct:.1%})"]


def plot_semantic_squares(
    df: pd.DataFrame,
    column: str = "semantic",
    top_n: int | None = None,
    figsize: tuple = (11, 7),
    title: str | None = None,
    fontsize: int = 10,
):
    """
    Draw a proportional-square chart for the value counts of `column`.
    Small squares get their label pushed out to the margin with a
    leader-line arrow instead of overlapping text crammed inside.

    Parameters
    ----------
    df : DataFrame containing the column to visualise.
    column : name of the column to summarise (default 'semantic').
    top_n : if set, keep only the top_n most frequent values and group
            everything else into an 'Other' square.
    figsize : matplotlib figure size.
    title : chart title; defaults to "Distribution of <column>".
    fontsize : base font size for labels.
    """
    counts = df[column].value_counts(dropna=False)

    if top_n is not None and len(counts) > top_n:
        top = counts.iloc[:top_n]
        other_total = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other_total})])

    total = counts.sum()
    proportions = counts / total

    names = list(counts.index.astype(str))
    values = counts.values.astype(float)

    # --- Layout the treemap in a 0-100 x 0-100 box ---
    PAD = 32  # margin reserved on each side for overflow labels
    sizes = squarify.normalize_sizes(values, 100, 100)
    rects = squarify.squarify(sizes, 0, 0, 100, 100)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-PAD, 100 + PAD)
    ax.set_ylim(-8, 108)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.canvas.draw()  # need a renderer to measure text extents
    renderer = fig.canvas.get_renderer()

    colors = (cm.tab20.colors * (len(rects) // 20 + 1))[: len(rects)]

    # Pixels-per-data-unit, used to compare box size against text size
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    px_per_unit_x = abs(p1[0] - p0[0])
    px_per_unit_y = abs(p1[1] - p0[1])

    overflow = []  # boxes whose label didn't fit -> (cx, cy, name, count, pct, color)

    for rect, name, count, pct, color in zip(rects, names, counts.values, proportions.values, colors):
        x, y, dx, dy = rect["x"], rect["y"], rect["dx"], rect["dy"]
        ax.add_patch(Rectangle((x, y), dx, dy, facecolor=color, edgecolor="white", linewidth=1.5))

        lines = _label_lines(name, count, pct)
        # Measure the widest line's rendered pixel width/height at this fontsize
        probe = ax.text(0, 0, "\n".join(lines), fontsize=fontsize, ha="center", va="center")
        bbox = probe.get_window_extent(renderer=renderer)
        probe.remove()

        text_w_px, text_h_px = bbox.width, bbox.height
        box_w_px, box_h_px = dx * px_per_unit_x, dy * px_per_unit_y

        fits = (text_w_px < box_w_px * 0.92) and (text_h_px < box_h_px * 0.85)

        if fits:
            ax.text(
                x + dx / 2, y + dy / 2, "\n".join(lines),
                ha="center", va="center", fontsize=fontsize, color="black",
            )
        else:
            overflow.append((x + dx / 2, y + dy / 2, name, count, pct, color))

    # --- Place overflow labels in the side margins with leader arrows ---
    if overflow:
        left = [o for o in overflow if o[0] < 50]
        right = [o for o in overflow if o[0] >= 50]

        for side, items in (("left", left), ("right", right)):
            items.sort(key=lambda o: -o[1])  # top to bottom, matches visual order
            n = len(items)
            if n == 0:
                continue
            top_y, bot_y = 100, 0
            step = (top_y - bot_y) / n
            label_x = -PAD + 4 if side == "left" else 100 + PAD - 4
            ha = "left" if side == "left" else "right"

            for i, (bx, by, name, count, pct, color) in enumerate(items):
                label_y = top_y - step * (i + 0.5)
                lines = _label_lines(name, count, pct)
                ax.annotate(
                    "\n".join(lines),
                    xy=(bx, by), xycoords="data",
                    xytext=(label_x, label_y), textcoords="data",
                    ha=ha, va="center", fontsize=fontsize - 1,
                    arrowprops=dict(
                        arrowstyle="-", color=color, lw=1.3,
                        shrinkA=0, shrinkB=4,
                        connectionstyle="arc3,rad=0.0",
                    ),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.0),
                )

    ax.set_title(title or f"Distribution of '{column}'", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig, ax
