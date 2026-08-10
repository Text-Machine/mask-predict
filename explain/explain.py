"""
Corrected/hardened version of `MaskedLMExplainer` from `explain.py`.

This does NOT change `explain.py` — it subclasses `MaskedLMExplainer` so it's
a drop-in replacement (`MaskedLMExplainerImproved(...)` instead of
`MaskedLMExplainer(...)`), while reusing everything about the original that
was already correct (`forward_func`, target tokenization, word aggregation).

Issues found in `explain.py` and fixed here
--------------------------------------------
1. **Baseline blanks out [CLS]/[SEP] instead of keeping them fixed.**
   `baseline_ids = torch.full_like(input_ids, pad_token_id)` overwrites every
   position — including CLS and SEP — with the PAD id. Only the mask
   position(s) are then restored to match the input. Standard practice for
   IG on transformers (see Captum's own BERT tutorial) is the opposite:
   keep special tokens identical between input and baseline, and only swap
   the *content* tokens for the reference id. Otherwise:
     - part of the attribution mass gets assigned to the CLS/SEP embedding
       "moving" from PAD to CLS/SEP along the interpolation path, silently
       discarded later by `drop_special=True` — so the reported per-token
       scores no longer sum to F(input) - F(baseline) (completeness axiom
       broken for what's actually reported).
     - the L2 norm used to normalize scores is computed over *all* positions
       (including this CLS/SEP noise) before special tokens are dropped,
       which scales down the real content-token scores.
   Fixed by only replacing non-special, non-mask positions with the
   reference token id; CLS/SEP embeddings are then identical in input and
   baseline, so their IG contribution is (numerically) ~0 by construction.

2. **`pick_device()` used but never imported in `explain.py`.**
   `self.device = device or pick_device()` references a name that only
   exists in `explain/tools.py` / the package's `__init__.py`, not in
   `explain.py`'s own namespace. Every current notebook happens to always
   pass `device=pick_device()` explicitly, so this has never actually fired,
   but `MaskedLMExplainer(model_name=...)` with no `device` would raise
   `NameError`. Fixed by importing `pick_device` and resolving it in this
   module before calling into the parent `__init__`.

3. **No convergence check.** `ig.attribute()` was called without
   `return_convergence_delta=True`, so there was no way to tell whether the
   IG approximation (50 steps by default, against a `log_softmax` over the
   full vocabulary) had actually converged for a given example. Fixed by
   requesting the delta, exposing it in the result dict as
   `"convergence_delta"`, and optionally warning when it exceeds
   `convergence_warn_threshold`. `n_steps` / `internal_batch_size` are now
   also configurable instead of hard-coded to Captum's defaults.

4. **Normalization order.** `attrs` was L2-normalized *before*
   `attrs[mask_pos] = 0.0`, so the (small but nonzero, due to IG's numerical
   approximation) mask-position value contributed to the scale used for
   every other token. Fixed by zeroing the mask position(s) first.

Not changed (considered correct, or a deliberate modeling choice worth
documenting rather than "fixing"):
- `forward_func`'s handling of multi-token targets: it sums log p(target
  subtoken_k | context) over the K masked positions in a single forward
  pass. This is a pseudo-log-likelihood approximation of the joint
  probability of the multi-token target (not the true joint, which would
  require sequential/autoregressive infilling) — standard for BERT-style
  cloze scoring, but worth knowing when interpreting attribution magnitude
  for multi-token targets vs. single-token ones.
- Using the PAD embedding as the reference/baseline for content tokens is a
  design choice (also used in Captum's own tutorial), not a bug — a
  zero-vector or average-embedding baseline would be a legitimate
  alternative but changes what "no information" means, not "correctness".
"""

import warnings

import torch

from .explain_deprc import MaskedLMExplainer
from .tools import pick_device


class MaskedLMExplainerImproved(MaskedLMExplainer):
    def __init__(
        self,
        model_name="bert-base-uncased",
        device=None,
        n_steps=50,
        internal_batch_size=None,
        convergence_warn_threshold=0.05,
        auto_increase_steps=True,
        max_n_steps=200,
        step_growth=2.0,
    ):
        # Resolve device here (in this module's own namespace, where
        # `pick_device` actually exists) before calling into the parent
        # `__init__`, so its buggy `device or pick_device()` fallback never
        # has to execute.
        if device is None:
            device = pick_device()
        super().__init__(model_name=model_name, device=device)

        # Reference/baseline token id for "no content" positions, matching
        # Captum's IG-for-BERT convention. Falls back to unk_token_id for
        # tokenizers without a dedicated pad token.
        self.ref_token_id = self.tokenizer.pad_token_id
        if self.ref_token_id is None:
            self.ref_token_id = self.tokenizer.unk_token_id
        if self.ref_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad_token nor an unk_token to use as "
                "the IG reference/baseline id."
            )

        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.convergence_warn_threshold = convergence_warn_threshold

        # If the convergence delta exceeds `convergence_warn_threshold`,
        # `_attribute_with_convergence` re-runs IG with n_steps multiplied by
        # `step_growth` (capped at `max_n_steps`) instead of just warning.
        # This trades runtime for accuracy: each retry is a full extra
        # `ig.attribute()` call, so a chronically-high-delta model/target
        # combination will consistently pay for every escalation step.
        self.auto_increase_steps = auto_increase_steps
        self.max_n_steps = max_n_steps
        self.step_growth = step_growth

    def _build_baseline_ids(self, input_ids, mask_pos):
        """
        Baseline identical to `input_ids` at [CLS]/[SEP]/mask position(s),
        and set to `self.ref_token_id` everywhere else (the content tokens
        we actually want attribution for).
        """
        keep_as_input = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id):
            if special_id is not None:
                keep_as_input |= input_ids == special_id
        keep_as_input[0, mask_pos] = True

        ref = torch.full_like(input_ids, self.ref_token_id)
        return torch.where(keep_as_input, input_ids, ref)

    def _attribute_with_convergence(self, emb, base, attention_mask, mpos, tid, valid, target):
        """
        Run `self.ig.attribute()`, escalating `n_steps` (up to `max_n_steps`,
        by a factor of `step_growth` each retry) while the convergence delta
        stays above `convergence_warn_threshold`. Returns whichever attempt
        had the lowest delta, plus the delta and n_steps actually used, and
        warns if it never got under threshold.
        """
        n_steps = self.n_steps
        best_attrs = best_delta = best_n_steps = None

        while True:
            attrs, delta = self.ig.attribute(
                inputs=emb,
                baselines=base,
                additional_forward_args=(attention_mask, mpos, tid, valid),
                n_steps=n_steps,
                internal_batch_size=self.internal_batch_size,
                return_convergence_delta=True,
            )
            delta_val = float(delta.abs().max().item())

            if best_delta is None or delta_val < best_delta:
                best_attrs, best_delta, best_n_steps = attrs, delta_val, n_steps

            converged = self.convergence_warn_threshold is None or best_delta <= self.convergence_warn_threshold
            can_retry = self.auto_increase_steps and n_steps < self.max_n_steps
            if converged or not can_retry:
                break
            n_steps = min(int(round(n_steps * self.step_growth)), self.max_n_steps)

        if self.convergence_warn_threshold is not None and best_delta > self.convergence_warn_threshold:
            warnings.warn(
                f"IG convergence delta={best_delta:.4f} still exceeds threshold "
                f"{self.convergence_warn_threshold} for target={target!r} "
                f"after n_steps={best_n_steps} (max_n_steps={self.max_n_steps}); "
                "attribution may be inaccurate. Consider raising max_n_steps or "
                "the base n_steps.",
                stacklevel=3,
            )

        return best_attrs, best_delta, best_n_steps

    def explain(
        self,
        texts,
        target_words_list,
        normalize=True,
        drop_special=True,
        return_word_scores=True,
        word_agg="mean",
        show_progress=True,
        progress_desc="Explaining",
    ):
        if len(texts) != len(target_words_list):
            raise ValueError("texts and target_words_list must have same length")

        all_results = []
        iterator = zip(texts, target_words_list)
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=len(texts), desc=progress_desc)

        for text, targets in iterator:
            sent_out = {}

            for target in targets:
                target_ids = self._target_to_token_ids(target)
                if len(target_ids) == 0:
                    sent_out[target] = {"skipped": True, "reason": "empty tokenization"}
                    continue

                text_k = self._expand_single_mask(text, len(target_ids))
                enc = self.tokenizer([text_k], return_tensors="pt", padding=True, truncation=True)
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)

                mask_pos = (input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=False).flatten()
                if mask_pos.numel() != len(target_ids):
                    sent_out[target] = {
                        "skipped": True,
                        "reason": f"mask count ({mask_pos.numel()}) != target token count ({len(target_ids)})"
                    }
                    continue

                emb = self.model.get_input_embeddings()(input_ids)

                # --- FIX 1: baseline keeps CLS/SEP/mask identical to input,
                # only content tokens move to the reference embedding.
                baseline_ids = self._build_baseline_ids(input_ids, mask_pos)
                base = self.model.get_input_embeddings()(baseline_ids)

                mpos = mask_pos.unsqueeze(0)
                tid = torch.tensor(target_ids, device=self.device).unsqueeze(0)
                valid = torch.ones_like(tid, dtype=torch.float32, device=self.device)

                # --- FIX 3: request convergence delta + expose n_steps;
                # escalate n_steps automatically if it doesn't converge.
                attrs, delta_val, n_steps_used = self._attribute_with_convergence(
                    emb, base, attention_mask, mpos, tid, valid, target
                )

                attrs = attrs.sum(dim=-1).squeeze(0)

                # --- FIX 4: zero the mask position(s) before normalizing,
                # not after, so it can't skew the scale.
                attrs[mask_pos] = 0.0
                if normalize:
                    attrs = attrs / attrs.norm().clamp_min(1e-12)

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                token_rows = []
                for tok, val, tid_ in zip(tokens, attrs.tolist(), input_ids[0].tolist()):
                    if drop_special and tid_ in {
                        self.tokenizer.cls_token_id,
                        self.tokenizer.sep_token_id,
                        self.tokenizer.pad_token_id,
                    }:
                        continue
                    token_rows.append((tok, float(val)))

                result_obj = {
                    "skipped": False,
                    "target_token_ids": target_ids,
                    "token_attributions": token_rows,
                    "convergence_delta": delta_val,
                    "n_steps_used": n_steps_used,
                }

                if return_word_scores:
                    result_obj["word_attributions"] = self._aggregate_tokens_to_words(
                        token_rows, agg=word_agg
                    )

                sent_out[target] = result_obj

            all_results.append(sent_out)

        return all_results
