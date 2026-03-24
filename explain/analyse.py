import numpy as np


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