"""Hybrid lexical and local-model matching of targets to DataFrame columns."""

import re
import threading
from typing import Any

import pandas as pd
from config import TARGET_EMBEDDING_MODEL, TARGET_RERANKER_MODEL


MIN_MATCH_SCORE = 0.80
"""Minimum candidate score required for automatic column selection."""

MIN_SCORE_MARGIN = 0.25
"""Minimum gap required between the best and second-best candidates."""

MIN_CANDIDATE_SCORE = 0.50
"""Minimum score for a column to be included as a meaningful candidate."""

MIN_SEMANTIC_MATCH_SCORE = 0.92
"""Minimum contextual similarity required for an automatic semantic match."""

MIN_SEMANTIC_SCORE_MARGIN = 0.01
"""Minimum lead over the next semantic candidate required for auto-selection."""

MIN_CANDIDATE_NAME_SIMILARITY = 0.85
"""Above this label similarity, retain clarification for near-duplicate columns."""

_SEPARATOR_PATTERN = re.compile(r"[_\W]+", re.UNICODE)
_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_MAX_RERANK_CANDIDATES = 3
_BINARY_REQUEST_TERMS = {
    "whether", "will", "does", "has", "have", "leave", "leaves",
    "churn", "churns", "exit", "exits",
}
_MULTI_CLASS_REQUEST_TERMS = {"which", "belongs", "category", "class", "type", "species"}
_MODEL_LOCK = threading.Lock()
_MODEL_COMPONENTS: tuple[Any, ...] | None = None
_MODEL_LOAD_FAILED = False

# Common stopwords and filler phrases to remove from natural language targets
_STOPWORDS = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with",
    "from", "is", "are", "was", "were", "be", "been", "being",
    "if", "whether", "will", "would", "should", "could", "can",
    "this", "that", "these", "those"
}

# Common predictive/descriptive patterns to normalize
_PREDICTIVE_PATTERNS = [
    (re.compile(r"\b(?:predict|predicting|prediction)\s+(?:the\s+)?"), ""),
    (re.compile(r"\b(?:whether|if)\s+"), ""),
    (re.compile(r"\bwill\s+"), ""),
    (re.compile(r"\b(?:price|value|cost)\s+of\s+(?:a\s+|an\s+)?"), ""),
    (re.compile(r"\b(?:number|count|amount)\s+of\s+"), ""),
]

def _normalize(value: Any) -> str:
    """Normalize a target or column name without changing the source value."""
    return _SEPARATOR_PATTERN.sub(" ", str(value).strip().lower()).strip()


def _extract_keywords(target: str) -> str:
    """Extract keywords from natural language target descriptions."""
    # Start with normalized text
    text = target.lower().strip()
    
    # Apply predictive patterns to simplify common phrases
    for pattern, replacement in _PREDICTIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    
    # Split into tokens and remove stopwords
    tokens = text.split()
    keywords = [token for token in tokens if token and token not in _STOPWORDS]
    
    # Return space-separated keywords
    return " ".join(keywords) if keywords else target


def _score_match(target: str, column: str) -> float:
    """Score exact, keyword-based, token-overlap, and token-containment matches."""
    if not target or not column:
        return 0.0
    
    # Exact match gets perfect score
    if target == column:
        return 1.0

    target_tokens = set(target.split())
    column_tokens = set(column.split())
    overlap = target_tokens.intersection(column_tokens)
    
    # No overlap
    if not overlap:
        return 0.0

    # Calculate bidirectional coverage
    target_coverage = len(overlap) / len(target_tokens)
    column_coverage = len(overlap) / len(column_tokens)
    
    # If all column tokens are matched (column is subset of target), high confidence
    if column_coverage == 1.0:
        return 0.95
    
    # Boost score if all target keywords are present in column
    if target_coverage == 1.0:
        # All target keywords matched - high confidence but not perfect
        return 0.95
    
    # Partial match - use the better coverage direction
    best_coverage = max(target_coverage, column_coverage)
    return 0.8 * best_coverage


def _load_local_models() -> tuple[Any, ...] | None:
    """Load both local scoring models once; never fetch weights during inference."""
    global _MODEL_COMPONENTS, _MODEL_LOAD_FAILED
    if _MODEL_COMPONENTS is not None:
        return _MODEL_COMPONENTS
    if _MODEL_LOAD_FAILED:
        return None

    with _MODEL_LOCK:
        if _MODEL_COMPONENTS is not None:
            return _MODEL_COMPONENTS
        if _MODEL_LOAD_FAILED:
            return None
        try:
            import torch
            from transformers import (
                AutoModel,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedding_tokenizer = AutoTokenizer.from_pretrained(
                TARGET_EMBEDDING_MODEL,
                local_files_only=True,
            )
            embedding_model = AutoModel.from_pretrained(
                TARGET_EMBEDDING_MODEL,
                local_files_only=True,
            ).to(device).eval()
            reranker_tokenizer = AutoTokenizer.from_pretrained(
                TARGET_RERANKER_MODEL,
                local_files_only=True,
            )
            reranker_model = AutoModelForSequenceClassification.from_pretrained(
                TARGET_RERANKER_MODEL,
                local_files_only=True,
            ).to(device).eval()
            _MODEL_COMPONENTS = (
                torch,
                device,
                embedding_tokenizer,
                embedding_model,
                reranker_tokenizer,
                reranker_model,
            )
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            _MODEL_LOAD_FAILED = True
            return None
    return _MODEL_COMPONENTS


def _safe_value(value: Any) -> str:
    """Return a short, stable representation for candidate context."""
    if isinstance(value, float):
        return format(value, ".5g")
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:48]


def _representative_values(series: pd.Series) -> list[str]:
    non_null = series.dropna()
    if non_null.empty:
        return []
    if pd.api.types.is_numeric_dtype(series) and non_null.nunique() > 8:
        values = non_null.quantile([0.25, 0.5, 0.75]).drop_duplicates().tolist()
    else:
        try:
            values = non_null.value_counts().head(3).index.tolist()
        except TypeError:
            values = non_null.astype(str).value_counts().head(3).index.tolist()
    return [_safe_value(value) for value in values[:3]]


def _candidate_description(column: Any, series: pd.Series) -> str:
    column_name = _normalize(column)
    data_type = str(series.dtype)
    cardinality = int(series.nunique(dropna=True))
    values = ", ".join(_representative_values(series)) or "none"
    return (
        f"A data column named {column_name} has data type {data_type}, "
        f"cardinality {cardinality}, and representative values {values}."
    )


def _structural_score(target: str, series: pd.Series) -> float:
    """Score generic target-shape compatibility without domain-specific vocabulary."""
    tokens = set(_WORD_PATTERN.findall(target.lower()))
    is_categorical = (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    cardinality = int(series.nunique(dropna=True))
    is_binary_like = cardinality == 2
    if not is_categorical:
        if tokens.intersection(_BINARY_REQUEST_TERMS) and is_binary_like:
            return 1.0
        return 0.75 if not tokens.intersection(_BINARY_REQUEST_TERMS | _MULTI_CLASS_REQUEST_TERMS) else 0.35
    if tokens.intersection(_BINARY_REQUEST_TERMS):
        return 1.0 if cardinality == 2 else 0.2
    if tokens.intersection(_MULTI_CLASS_REQUEST_TERMS):
        return 1.0 if cardinality > 2 else 0.45
    return 0.65


def _apply_structural_preference(
    target_reference: str,
    dataframe: pd.DataFrame,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer target shapes compatible with explicit binary or multiclass wording."""
    tokens = set(_WORD_PATTERN.findall(target_reference.lower()))
    requests_binary = bool(tokens.intersection(_BINARY_REQUEST_TERMS))
    requests_multiclass = bool(tokens.intersection(_MULTI_CLASS_REQUEST_TERMS))
    if requests_binary == requests_multiclass or not candidates:
        return candidates

    compatible_candidates = [
        candidate
        for candidate in candidates
        if _structural_score(
            target_reference, dataframe[candidate["column"]]
        ) >= 0.95
    ]
    if not compatible_candidates or candidates[0] in compatible_candidates:
        return candidates

    best_compatible = max(compatible_candidates, key=lambda item: item["score"])
    best_compatible["score"] = max(
        best_compatible["score"], MIN_SEMANTIC_MATCH_SCORE
    )
    incompatible_score_cap = (
        MIN_SEMANTIC_MATCH_SCORE - MIN_SEMANTIC_SCORE_MARGIN - 1e-6
    )
    for candidate in candidates:
        if candidate not in compatible_candidates:
            candidate["score"] = min(candidate["score"], incompatible_score_cap)
    candidates.sort(key=lambda row: (-row["score"], row["column"]))
    return candidates


def _hybrid_candidate_scores(
    target_reference: str,
    dataframe: pd.DataFrame,
    columns: list[Any],
    lexical_scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Retrieve candidates by complete-phrase embeddings, then cross-encode top three."""
    if len(_WORD_PATTERN.findall(target_reference)) < 2:
        return []
    components = _load_local_models()
    if components is None or not columns:
        return []

    torch, device, embedding_tokenizer, embedding_model, reranker_tokenizer, reranker_model = components
    descriptions = {
        str(column): _candidate_description(column, dataframe[column])
        for column in columns
    }
    labels = {
        str(column): f"Column name: {_normalize(column)}."
        for column in columns
    }
    normalized_target = str(target_reference).strip()
    try:
        inputs = embedding_tokenizer(
            [normalized_target, *descriptions.values(), *labels.values()],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            hidden = embedding_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            count = len(descriptions)
            retrieval_scores = torch.mv(pooled[1:count + 1], pooled[0]).cpu().tolist()
            label_vectors = pooled[count + 1:]
    except (RuntimeError, TypeError, ValueError):
        return []

    retrieval_rows = []
    for column, score in zip(descriptions, retrieval_scores):
        lexical_score = float(lexical_scores.get(column, 0.0))
        structural_score = _structural_score(target_reference, dataframe[column])
        retrieval_confidence = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
        retrieval_rank_score = (
            0.85 * retrieval_confidence
            + 0.15 * lexical_score
            if lexical_score >= MIN_CANDIDATE_SCORE
            else retrieval_confidence
        )
        retrieval_rows.append(
            {
                "column": column,
                "embedding_score": float(score),
                "retrieval_confidence": retrieval_confidence,
                "lexical_score": lexical_score,
                "structural_score": structural_score,
                "retrieval_rank_score": retrieval_rank_score,
            }
        )
    retrieval_rows.sort(
        key=lambda row: (-row["retrieval_rank_score"], row["column"])
    )
    shortlist = retrieval_rows[:_MAX_RERANK_CANDIDATES]
    candidate_labels = [labels[row["column"]] for row in shortlist]
    hypothesis = normalized_target

    try:
        reranker_inputs = reranker_tokenizer(
            [hypothesis] * len(shortlist),
            candidate_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        reranker_inputs = {
            name: value.to(device) for name, value in reranker_inputs.items()
        }
        with torch.no_grad():
            logits = reranker_model(**reranker_inputs).logits.reshape(-1)
            reranker_support = torch.softmax(logits / 0.01, dim=0)
            embedding_logits = torch.tensor(
                [row["embedding_score"] for row in shortlist],
                dtype=torch.float32,
                device=device,
            )
            embedding_support = torch.softmax(embedding_logits / 0.08, dim=0)
            lexical_logits = torch.tensor(
                [row["lexical_score"] for row in shortlist],
                dtype=torch.float32,
                device=device,
            )
            has_lexical_evidence = any(
                row["lexical_score"] >= MIN_CANDIDATE_SCORE for row in shortlist
            )
            if has_lexical_evidence:
                lexical_support = torch.softmax(lexical_logits / 0.15, dim=0)
                combined_support = (
                    0.03 * embedding_support
                    + 0.92 * reranker_support
                    + 0.05 * lexical_support
                )
            else:
                combined_support = 0.03 * embedding_support + 0.97 * reranker_support
            reranker_scores = reranker_support.cpu().tolist()
            final_scores = combined_support.cpu().tolist()
    except (RuntimeError, TypeError, ValueError):
        return []

    reranked = []
    for row, reranker_score, final_score in zip(shortlist, reranker_scores, final_scores):
        row["reranker_score"] = float(reranker_score)
        if row["structural_score"] >= 0.95:
            row["score"] = float(
                min(1.0, 0.90
                + 0.20 * (
                    0.60 * row["retrieval_confidence"]
                    + 0.40 * row["reranker_score"]
                ))
            )
        else:
            row["score"] = float(0.75 * final_score + 0.25 * row["structural_score"])
        reranked.append(row)
    reranked.sort(key=lambda row: (-row["score"], row["column"]))
    reranked = _apply_structural_preference(
        target_reference, dataframe, reranked
    )
    if len(reranked) > 1:
        label_index = {column: index for index, column in enumerate(labels)}
        top_index = label_index[reranked[0]["column"]]
        runner_index = label_index[reranked[1]["column"]]
        name_similarity = float(
            torch.dot(label_vectors[top_index], label_vectors[runner_index]).item()
        )
        reranked[0]["runner_up_name_similarity"] = name_similarity
        if name_similarity >= MIN_CANDIDATE_NAME_SIMILARITY:
            ambiguity_score = MIN_SEMANTIC_MATCH_SCORE - 1e-6
            reranked[0]["score"] = min(reranked[0]["score"], ambiguity_score)
            reranked[1]["score"] = min(reranked[1]["score"], ambiguity_score)
            reranked.sort(key=lambda row: (-row["score"], row["column"]))
    return reranked


def _build_result(
    target_reference: str,
    candidates: list[dict[str, Any]],
    *,
    minimum_score: float,
    minimum_margin: float,
    match_type: str,
) -> dict[str, Any]:
    if not candidates:
        return {
            "target_reference": target_reference,
            "matched_column": None,
            "confidence": 0.0,
            "score": 0.0,
            "runner_up": None,
            "margin": 0.0,
            "candidates": [],
            "needs_clarification": True,
        }

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    second_score = runner_up["score"] if runner_up is not None else None
    margin = (
        float(best["score"] - second_score)
        if second_score is not None
        else (0.0 if match_type == "hybrid" else float(best["score"]))
    )
    is_clear_match = (
        float(best["score"]) >= minimum_score
        and margin >= minimum_margin
    )
    return {
        "target_reference": target_reference,
        "matched_column": best["column"] if is_clear_match else None,
        "confidence": float(best["score"]) if is_clear_match else 0.0,
        "score": float(best["score"]),
        "runner_up": (
            {"column": runner_up["column"], "score": float(second_score)}
            if runner_up is not None
            else None
        ),
        "margin": margin,
        "match_type": match_type,
        "candidates": candidates,
        "needs_clarification": not is_clear_match,
    }


def match_target_column(
    target_reference: str,
    df: pd.DataFrame,
    *,
    candidate_columns: list[str] | None = None,
    require_exact_match: bool = False,
) -> dict[str, Any]:
    """Match a natural-language target reference to a DataFrame column."""
    if not isinstance(target_reference, str):
        raise TypeError("target_reference must be a string")
    if not target_reference.strip():
        raise ValueError("target_reference must be a non-empty string")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(df.columns) == 0:
        raise ValueError("df must contain at least one column")

    normalized_target = _normalize(target_reference)
    keyword_target = _extract_keywords(normalized_target)
    allowed_columns = (
        {str(column) for column in candidate_columns}
        if candidate_columns is not None
        else None
    )
    columns_to_match = [
        column for column in df.columns
        if allowed_columns is None or str(column) in allowed_columns
    ]
    if require_exact_match:
        columns_to_match = [
            column
            for column in columns_to_match
            if _normalize(column) == normalized_target
        ]
    if not columns_to_match:
        return _build_result(
            target_reference,
            [],
            minimum_score=MIN_MATCH_SCORE,
            minimum_margin=MIN_SCORE_MARGIN,
            match_type="lexical",
        )
    
    scored_candidates: list[dict[str, Any]] = []
    lexical_scores: dict[str, float] = {}
    for original_column in columns_to_match:
        normalized_column = _normalize(original_column)
        
        # Try both normalized and keyword-extracted versions
        score_normalized = _score_match(normalized_target, normalized_column)
        score_keywords = _score_match(keyword_target, normalized_column)
        score = max(score_normalized, score_keywords)
        lexical_scores[str(original_column)] = float(score)
        
        if score >= MIN_CANDIDATE_SCORE:
            scored_candidates.append(
                {"column": str(original_column), "score": float(score)}
            )

    lexical_candidates = sorted(
        scored_candidates,
        key=lambda candidate: (-candidate["score"], candidate["column"]),
    )
    lexical_second = lexical_candidates[1]["score"] if len(lexical_candidates) > 1 else 0.0
    if (
        lexical_candidates
        and lexical_candidates[0]["score"] >= MIN_MATCH_SCORE
        and lexical_candidates[0]["score"] - lexical_second >= MIN_SCORE_MARGIN
    ):
        return _build_result(
            target_reference,
            lexical_candidates,
            minimum_score=MIN_MATCH_SCORE,
            minimum_margin=MIN_SCORE_MARGIN,
            match_type="lexical",
        )
    if lexical_candidates and lexical_candidates[0]["score"] >= MIN_MATCH_SCORE:
        return _build_result(
            target_reference,
            lexical_candidates,
            minimum_score=MIN_MATCH_SCORE,
            minimum_margin=MIN_SCORE_MARGIN,
            match_type="lexical",
        )
    hybrid_candidates = _hybrid_candidate_scores(
        normalized_target,
        df,
        columns_to_match,
        lexical_scores,
    )
    if hybrid_candidates:
        return _build_result(
            target_reference,
            hybrid_candidates,
            minimum_score=MIN_SEMANTIC_MATCH_SCORE,
            minimum_margin=MIN_SEMANTIC_SCORE_MARGIN,
            match_type="hybrid",
        )

    return _build_result(
        target_reference,
        lexical_candidates,
        minimum_score=MIN_MATCH_SCORE,
        minimum_margin=MIN_SCORE_MARGIN,
        match_type="lexical",
    )
