# Installation In Instructions

Clone the repo.

```bash
!git clone https://github.com/Text-Machine/mask-predict.git
```

Move to the folder.

```bash
%cd mask-predict
```

Install the module.

```bash
!pip install -e .
```

Look at the `compute_influence_improved.ipynb` notebook for example usage of the code.

# How Integrated Gradients Works in This Notebook

## Overview

This notebook uses **Integrated Gradients (IG)** to estimate how much each context word contributes to the model's score for a chosen masked-token prediction.

## 1. What problem IG solves

A masked language model gives a score for a target token at the `[MASK]` position.

For example, for:

> `the cotton was picked by [MASK]`

the model may assign some score or probability to `"slaves"`.

The question is:

- **Which surrounding words pushed the model toward that prediction?**
- **Which words pushed against it?**

IG answers this by assigning an **attribution score** to each input token.

---

## 2. Core idea of Integrated Gradients

A plain gradient asks:

- *If I make the input infinitesimally different right now, how does the output change?*

That is often noisy or unstable.

Integrated Gradients instead asks:

- *What happens if I move gradually from a neutral baseline input to the real input, and accumulate the gradients along the way?*

So instead of looking at the model only at the final sentence representation, IG traces a path:

- **baseline input** → **real input**

and integrates the gradient over that path.

---

## 3. Mathematical formulation

If $x$ is the real input embedding and $x'$ is the baseline embedding, then the attribution for input dimension $i$ is:

$$IG_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$

Where:

- $F(\cdot)$ is the model output being explained
- $x'$ is the baseline
- $x$ is the actual input
- $\alpha$ moves from 0 to 1 along the path

In words:

1. Start from baseline embeddings
2. Interpolate toward the true embeddings
3. Measure the gradient of the target score at many points
4. Accumulate those gradients
5. Scale by the difference between real input and baseline

---

## 4. What is the "input" in this notebook?

Because the model input tokens are **discrete IDs**, gradients cannot be taken directly with respect to token IDs.

So the notebook explains the model using **input embeddings** instead:

```python
emb = self.model.get_input_embeddings()(input_ids)
base = self.model.get_input_embeddings()(baseline_ids)
```

This is standard for transformer attribution.

- `emb` = embedding vectors for the actual sentence
- `base` = embedding vectors for a neutral baseline sentence

---

## 5. What is the baseline here?

The notebook builds a baseline made mostly of **PAD tokens**, while keeping the mask positions fixed:

```python
baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
baseline_ids[0, mask_pos] = self.tokenizer.mask_token_id
```

This means:

- the model still sees the same masked location(s)
- most contextual information is removed
- the sentence is reduced to something like a neutral "empty context"

So IG measures:

- how much each real context word matters **relative to this almost content-free version**

This is why the explanation can be interpreted as contextual influence.

---

## 6. What output is being explained?

In this notebook, the function being explained is not the whole model output. It is a **specific target score**.

For single- or multi-token targets, the code computes:

- logits at the mask positions
- log-probabilities over the vocabulary
- the log-probability of the selected target token(s)

This happens in `forward_func`:

```python
def forward_func(self, inputs_embeds, attention_mask, mask_indices, target_indices, valid_positions):
    logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
    bsz = logits.size(0)

    # (B, K, V): logits at each masked position
    mask_logits = logits[
        torch.arange(bsz, device=logits.device).unsqueeze(1),
        mask_indices
    ]
    log_probs = torch.log_softmax(mask_logits, dim=-1)  # (B, K, V)

    # (B, K): log p(target_token_k | context)
    target_lp = log_probs.gather(-1, target_indices.unsqueeze(-1)).squeeze(-1)

    # Sum only valid positions
    return (target_lp * valid_positions).sum(dim=1)  # (B,)
```

### Interpretation

`forward_func(...)` returns:

- the **log-probability** of the chosen target token if there is one mask
- or the **sum of log-probabilities** of all target subword pieces if there are multiple masks

So IG explains:

> **Which context tokens most increase or decrease the model's log-probability for this target?**

---

## 7. Why use log-probabilities?

Using `log_softmax` is often preferable to raw probabilities because:

- probabilities can saturate
- gradients become very small near extremes
- log-probabilities behave more smoothly for attribution

So this line is important:

```python
log_probs = torch.log_softmax(mask_logits, dim=-1)
```

It means the explanation is tied to the model's **preference strength** for the target.

---

## 8. How Captum computes the integral

The notebook uses Captum's `IntegratedGradients`:

```python
self.ig = IntegratedGradients(self.forward_func)
```

and later:

```python
attrs = self.ig.attribute(
    inputs=emb,
    baselines=base,
    additional_forward_args=(attention_mask, mpos, tid, valid),
).sum(dim=-1).squeeze(0)
```

Captum does not compute the integral analytically. It approximates it numerically by:

1. creating many interpolated inputs between `base` and `emb`
2. running the model on each interpolated input
3. computing gradients of the target score
4. summing them to approximate the path integral

So conceptually it evaluates a sequence like:

$$x' ,\; x' + \tfrac{1}{n}(x-x'),\; x' + \tfrac{2}{n}(x-x'),\; \dots,\; x$$

and accumulates gradients along the way.

---

## 9. Why `.sum(dim=-1)`?

The output of IG is an attribution for **every embedding dimension** of every token.

So initially the shape is roughly:

- `(sequence_length, embedding_dim)`

But humans want one score per token. So the code sums across embedding dimensions:

```python
attrs = self.ig.attribute(...).sum(dim=-1).squeeze(0)
```

This produces one scalar attribution per token.

### Interpretation

- **positive score**: token supports the target prediction
- **negative score**: token suppresses the target prediction
- **near zero**: token has little effect relative to the baseline

---

## 10. Why are mask-token attributions zeroed out?

After attribution, the code removes the contribution of the mask positions:

```python
attrs[mask_pos] = 0.0
```

This is done because the goal is to measure the effect of the **context words**, not the trivial contribution of the `[MASK]` token itself.

---

## 11. Word-level aggregation

Because BERT uses **subword tokenization**, one visible word may be split into pieces:

- `working` → `work`, `##ing`
- `industrialization` → several subwords

The notebook first computes token-level attributions, then aggregates them into word-level scores:

```python
result_obj["word_attributions"] = self._aggregate_tokens_to_words(
    token_rows, agg=word_agg
)
```

The aggregation method is controlled by the `agg` parameter:

- `agg="mean"`: average the subword scores
- `agg="max"`: take the maximum subword score

This makes the explanation easier to interpret at the level of actual words.

---

## 12. What the attribution means in practice

Suppose the sentence is:

> `the plantation relied on [MASK] labour`

and the target is `"slave"`.

If IG gives high positive attribution to:

- `plantation`
- `labour`

that means these words strongly increase the model's score for `"slave"` relative to the baseline.

If another word gets a negative score, it means that word pushes the model away from that target.

So the method measures **contextual support** for the target prediction.

---

## 13. Important caveat

IG is not measuring causal truth in a strict experimental sense. It measures **model sensitivity along a chosen path from baseline to input**.

So results depend on:

- the chosen baseline
- the target score being explained
- the tokenizer and subword segmentation
- whether scores are normalized afterward

In this code, attribution vectors are normalized:

```python
if normalize:
    attrs = attrs / attrs.norm().clamp_min(1e-12)
```

This is useful for comparing relative importance within a sentence, but it changes the original magnitude. So after normalization, scores are best interpreted as **relative influence**, not absolute effect size.

---

## 14. Multi-token targets

The notebook extends IG to handle multi-token targets (e.g., `"steam engine"`).

For a multi-token target:

1. The single `[MASK]` is expanded to multiple masks:

```python
text_k = self._expand_single_mask(text, len(target_ids))
```

2. The target score is the **sum of log-probabilities**:

```python
return (target_lp * valid_positions).sum(dim=1)
```

This gives an attribution that explains:

> **Which context tokens increase or decrease the model's joint probability for this multi-token sequence?**

---

## 15. Comparing two models

The notebook includes a `compare_explainers` function that runs two models and computes per-word differences:

```python
comparison = compare_explainers(explainer_new, explainer_old, texts, single_target, level="word", word_agg="max")
```

For each word, the output is a tuple:

```
(word, score_model1, score_model2, difference)
```

This allows you to see:

- which words became more or less influential between time periods
- how model behavior changed historically

---

## 16. Summary

In this notebook, Integrated Gradients works as follows:

1. Build a **baseline sentence embedding** with little contextual content
2. Build the **real sentence embedding**
3. Define a scalar output: the model's **log-probability for a chosen masked-token target**
4. Interpolate from baseline to real input
5. Accumulate gradients of the target score along that path
6. Sum over embedding dimensions to get **one attribution per token**
7. Aggregate subwords to get **one attribution per word**

The final word scores estimate:

> **How much each context word contributes to the model predicting the chosen masked target.**

This provides an interpretable, model-agnostic explanation of masked language model predictions.