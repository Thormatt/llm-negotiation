"""
LLMIO for the negotiation domain.

Translates a natural-language preference description into a 5-dim weight
vector on the simplex. This is the I/O layer that sits between a human user
and the middleman protocol. The LLM does *translation only* — never strategy.

The intent string is a structured-but-fuzzy preference description with
explicit issue names and qualitative priority levels. The LLM's job is to
output a JSON weight vector that reflects the description's relative
priorities.

Translation accuracy is measured downstream by comparing LLM-output weights
against the ground-truth weights that generated the description.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


VARIABLES = ["price", "contract_length", "payment_terms", "sla_tier", "data_rights"]


@dataclass
class IntentTranslation:
    weights: np.ndarray              # (5,) simplex
    raw_response: str
    error: Optional[str] = None


def render_preferences(weights: np.ndarray, role: str,
                       include_numeric_weights: bool = False) -> str:
    """Render a 5-dim weight vector as a natural-language preference description.

    By default the description uses *qualitative* importance labels only
    (MOST important / very important / moderately important / less important
    / least important) — translating this back into a numeric weight vector
    requires real interpretation, and that is the test we want.

    `include_numeric_weights=True` includes the exact numeric weight in each
    line. Useful as a fidelity-baseline ablation (translation collapses to
    pure copy-paste).
    """
    if weights.shape != (5,):
        raise ValueError("expected weight vector of length 5")

    direction_words = {
        "buyer": {
            "price": "LOW",
            "contract_length": "LONG",
            "payment_terms": "SLOW (delayed)",
            "sla_tier": "HIGH",
            "data_rights": "CLIENT-favoured",
        },
        "seller": {
            "price": "HIGH",
            "contract_length": "SHORT",
            "payment_terms": "FAST (upfront)",
            "sla_tier": "LOW",
            "data_rights": "VENDOR-favoured",
        },
    }[role]

    order = np.argsort(-weights)  # descending
    levels = ["MOST important", "very important", "moderately important",
              "less important", "least important"]

    lines = [f"You are the {role.upper()}. Your priorities for the 5 contract issues are:"]
    for rank, idx in enumerate(order):
        v = VARIABLES[idx]
        if include_numeric_weights:
            lines.append(
                f"  - {v}: {levels[rank]} (you want {direction_words[v]}; weight = {weights[idx]:.3f})"
            )
        else:
            lines.append(
                f"  - {v}: {levels[rank]}, you want {direction_words[v]}"
            )
    lines.append(
        "\nTranslate these qualitative priorities into a 5-dim utility weight vector "
        "summing to 1.0. The MOST important issue should have the highest weight, "
        "and so on, with sensible spacing across the qualitative levels."
    )
    return "\n".join(lines)


def translate_intent(client, model: str, intent: str,
                     temperature: float = 0.0, max_tokens: int = 600) -> IntentTranslation:
    """Translate a preference description into a weight vector via LLM.

    The LLM is asked to output a JSON dict of weights summing to 1, in the
    fixed VARIABLE order. We don't ask it to reason about strategy — only to
    parse the description and emit the corresponding numeric vector.
    """
    system = (
        "You are a translation layer. Read the user's preference description "
        "and output their utility weights as a JSON object. The weights must "
        "sum to approximately 1.0 (we will normalize). Higher weight = more "
        "important. Output JSON only, no commentary.\n\n"
        f"The 5 issues are: {VARIABLES}.\n\n"
        'Output format:  {"weights": {"price": 0.32, "contract_length": 0.18, '
        '"payment_terms": 0.21, "sla_tier": 0.05, "data_rights": 0.24}}'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": intent},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return IntentTranslation(
            weights=np.full(5, 0.2, dtype=np.float64),
            raw_response="",
            error=str(e),
        )

    weights = _parse_weights(raw)
    if weights is None:
        return IntentTranslation(
            weights=np.full(5, 0.2, dtype=np.float64),
            raw_response=raw,
            error="parse_failure",
        )
    return IntentTranslation(weights=weights, raw_response=raw)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_weights(raw: str) -> Optional[np.ndarray]:
    text = raw
    if "```" in text:
        chunks = text.split("```")
        if len(chunks) >= 2:
            text = chunks[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    if not text.startswith("{"):
        m = _JSON_RE.search(raw)
        if m:
            text = m.group(0)

    try:
        decision = json.loads(text, strict=False)
        weights_dict = decision.get("weights", decision)
        if not isinstance(weights_dict, dict):
            return None
        vec = np.array([float(weights_dict.get(v, 0.0)) for v in VARIABLES],
                       dtype=np.float64)
        if vec.sum() <= 0 or np.any(np.isnan(vec)):
            return None
        return vec / vec.sum()
    except Exception:
        return None
