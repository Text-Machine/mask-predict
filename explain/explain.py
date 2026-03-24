import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from captum.attr import IntegratedGradients

class MaskedLMExplainer:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.ig = IntegratedGradients(self.forward_func)

    # Supports K mask positions / K target tokens per example
    def forward_func(self, inputs_embeds, attention_mask, mask_indices, target_indices, valid_positions):
        logits = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits  # (B, L, V)
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

    def _target_to_token_ids(self, text_target):
        ids = self.tokenizer(text_target, add_special_tokens=False)["input_ids"]
        return ids

    def _expand_single_mask(self, text, n_masks):
        mask = self.tokenizer.mask_token
        if text.count(mask) != 1:
            raise ValueError("Input sentence must contain exactly one [MASK] before expansion.")
        return text.replace(mask, " ".join([mask] * n_masks), 1)
    
    def _aggregate_tokens_to_words(self, token_rows, agg="mean"):
        """
        token_rows: list[(token, score)]
        Returns: list[(word, aggregated_score)]
        """
        if agg not in {"mean", "max"}:
            raise ValueError("agg must be 'mean' or 'max'")

        def reduce_scores(scores):
            return float(sum(scores) / len(scores)) if agg == "mean" else float(max(scores))

        word_rows = []
        current_word = None
        current_scores = []

        for tok, score in token_rows:
            # BERT WordPiece continuation
            if tok.startswith("##"):
                piece = tok[2:]
                if current_word is None:
                    current_word = piece
                    current_scores = [score]
                else:
                    current_word += piece
                    current_scores.append(score)
            else:
                if current_word is not None:
                    word_rows.append((current_word, reduce_scores(current_scores)))
                current_word = tok
                current_scores = [score]

        if current_word is not None:
            word_rows.append((current_word, reduce_scores(current_scores)))

        return word_rows

    def explain(
        self,
        texts,
        target_words_list,
        normalize=True,
        drop_special=True,
        return_word_scores=True,
        word_agg="mean",  # "mean" or "max"
    ):
        if len(texts) != len(target_words_list):
            raise ValueError("texts and target_words_list must have same length")

        all_results = []
        for text, targets in zip(texts, target_words_list):
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
                baseline_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
                baseline_ids[0, mask_pos] = self.tokenizer.mask_token_id
                base = self.model.get_input_embeddings()(baseline_ids)

                mpos = mask_pos.unsqueeze(0)
                tid = torch.tensor(target_ids, device=self.device).unsqueeze(0)
                valid = torch.ones_like(tid, dtype=torch.float32, device=self.device)

                attrs = self.ig.attribute(
                    inputs=emb,
                    baselines=base,
                    additional_forward_args=(attention_mask, mpos, tid, valid),
                ).sum(dim=-1).squeeze(0)

                if normalize:
                    attrs = attrs / attrs.norm().clamp_min(1e-12)

                attrs[mask_pos] = 0.0

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
                }

                if return_word_scores:
                    result_obj["word_attributions"] = self._aggregate_tokens_to_words(
                        token_rows, agg=word_agg
                    )

                sent_out[target] = result_obj

            all_results.append(sent_out)

        return all_results


def compare_explainers(explainer_1, explainer_2, texts, targets, level="word", word_agg="mean"):
    r1 = explainer_1.explain(texts, targets, return_word_scores=(level == "word"), word_agg=word_agg)
    r2 = explainer_2.explain(texts, targets, return_word_scores=(level == "word"), word_agg=word_agg)

    key = "word_attributions" if level == "word" else "token_attributions"

    comparisons = []
    for i in range(len(texts)):
        out = {}
        for target in targets[i]:
            a = r1[i][target]
            b = r2[i][target]
            if a.get("skipped") or b.get("skipped"):
                out[target] = {"skipped": True, "model1": a, "model2": b}
                continue

            rows1 = a[key]
            rows2 = b[key]

            t1 = [x[0] for x in rows1]
            t2 = [x[0] for x in rows2]
            if t1 != t2:
                raise ValueError(f"{level.capitalize()} mismatch at sentence {i}, target '{target}'")

            s1 = [x[1] for x in rows1]
            s2 = [x[1] for x in rows2]
            out[target] = list(zip(t1, s1, s2, [x - y for x, y in zip(s1, s2)]))
        comparisons.append(out)
    return comparisons
