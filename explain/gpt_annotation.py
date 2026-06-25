"""
gpt_preannotation.py
~~~~~~~~~~~~~~~~~~~~~
OpenAI-based chain-of-thought pre-annotation for newspaper match dataframes.

Functions
---------
run_gpt_preannotation(frame, system_prompt, ...)
    Pre-annotate unlabelled rows using the OpenAI chat API.
    Writes ``gpt_label``, ``gpt_reasoning``, and ``gpt_error`` columns back into
    *frame* in-place and returns a summary dict.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


# ── internal helper ────────────────────────────────────────────────────────────

def _call_openai_cot(
    client: OpenAI,
    model: str,
    system_prompt: str,
    snippet_text: str,
) -> tuple[str, str, str]:
    """
    Send one snippet to the OpenAI chat API with CoT instructions.

    Returns
    -------
    (label, reasoning, error)
        ``label`` is ``'yes'``, ``'no'``, or ``''`` on parse failure.
        ``error`` is a non-empty string when an exception occurred.
    """
    user_msg = f"Snippet:\n{snippet_text}"
    raw = ""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().lower()
        # if label not in ("yes", "no"):
        #     label = ""
        reasoning = str(parsed.get("reasoning", "")).strip()
        return label, reasoning, ""
    except json.JSONDecodeError as exc:
        return "", "", f"JSON parse error: {exc} | raw: {raw[:200]}"
    except Exception as exc:  # noqa: BLE001
        return "", "", str(exc)


# ── public API ─────────────────────────────────────────────────────────────────

def run_gpt_preannotation(
    frame,
    system_prompt: str,
    api_key: str = "",
    model: str = "gpt-4o-mini",
    n: int = 100,
    max_workers: int = 5,
) -> dict:
    """
    Pre-annotate *n* unlabelled rows using the OpenAI chat API with CoT prompting.

    Results are written back into *frame* in-place:

    - ``frame['gpt_label']``     — ``'yes'`` or ``'no'``
    - ``frame['gpt_reasoning']`` — the model's chain-of-thought reasoning
    - ``frame['gpt_error']``     — non-empty string if the API call failed

    Parameters
    ----------
    frame : pandas.DataFrame
        Must contain a ``'snippet'`` column and the three GPT columns above
        (created by the notebook data-load cell).
    system_prompt : str
        The system prompt sent to the model.  Should instruct it to reply with
        ``{"reasoning": "...", "label": "yes" | "no"}``.
    api_key : str
        OpenAI API key.  If blank, falls back to the ``OPENAI_API_KEY``
        environment variable.
    model : str
        Any OpenAI chat model name.
    n : int
        Maximum number of unlabelled rows to annotate in this run.
        Set to ``0`` to annotate all unlabelled rows.
    max_workers : int
        Number of parallel API request threads.

    Returns
    -------
    dict
        ``{"sent": int, "ok": int, "errors": int}``
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "No OpenAI API key found.  Set OPENAI_API_KEY in the settings cell "
            "or export it as an environment variable."
        )
    client = OpenAI(api_key=key)

    unlabelled_mask = frame["label"].fillna("").astype(str).str.strip() == ""
    candidate_idx = frame[unlabelled_mask].index.tolist()
    if n > 0:
        candidate_idx = candidate_idx[:n]

    if not candidate_idx:
        print("No unlabelled rows to pre-annotate.")
        return {"sent": 0, "ok": 0, "errors": 0}

    print(f"Sending {len(candidate_idx)} snippets to {model} …")

    def _worker(idx):
        snippet_text = str(frame.at[idx, "sentence"]).strip()
        return idx, _call_openai_cot(client, model, system_prompt, snippet_text)

    ok_count = 0
    err_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, idx): idx for idx in candidate_idx}
        for i, fut in enumerate(futures):
            idx, (label, reasoning, error) = fut.result()
            
            frame.at[idx, "label"] = label
            frame.at[idx, "reasoning"] = reasoning
            frame.at[idx, "gpt_error"] = error
            if error:
                err_count += 1
            else:
                ok_count += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(candidate_idx):
                print(f"  {i + 1}/{len(candidate_idx)} done  (ok={ok_count}, errors={err_count})")

    summary = {"sent": len(candidate_idx), "ok": ok_count, "errors": err_count}
    print(f"\nPre-annotation complete: {summary}")
    return summary
