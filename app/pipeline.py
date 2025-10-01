from typing import List, Dict, Any, Optional
import functools

import numpy as np
import langid
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer


# -------- Translation --------
@functools.lru_cache(maxsize=1)
def _get_translation_models():
    """Returns (model, tokenizer) if available, else (None, None).

    Gracefully handles missing optional dependencies like SentencePiece by
    skipping translation instead of raising runtime errors.
    """
    try:
        model_name = "Helsinki-NLP/opus-mt-mul-en"
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception:
        # Optional translation backend not available; fall back to no-translate
        return None, None


def translate_to_english(text: str) -> str:
    model, tokenizer = _get_translation_models()
    if model is None or tokenizer is None:
        return text
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=256)
    translated = model.generate(inputs, num_beams=4, max_length=256)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text


def translate_to_english_if_needed(text: str, language_hint: Optional[str] = None) -> str:
    if not text:
        return text

    lang = None
    if language_hint and language_hint.lower() != "auto":
        lang = language_hint.lower()
    else:
        # Detect language automatically
        lang, _ = langid.classify(text)

    if lang and lang.startswith("en"):
        return text
    return translate_to_english(text)


# -------- Embeddings --------
@functools.lru_cache(maxsize=1)
def _get_embedding_model():
    # `all-MiniLM-L6-v2` is compact and strong for semantic similarity
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_template_embeddings(templates: List[Dict[str, Any]]) -> np.ndarray:
    model = _get_embedding_model()
    descriptions = [template["description"] for template in templates]
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


# -------- Similarity & Search --------
def _cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # query_vec: (D,), matrix: (N, D) -> returns (N,)
    # Assumes rows in matrix are already L2-normalized; normalize query
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    q = query_vec / norm
    return matrix @ q


def semantic_search(
    query_text: str,
    templates: List[Dict[str, Any]],
    template_embeddings: np.ndarray,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    model = _get_embedding_model()
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
    scores = _cosine_similarity_matrix(query_embedding, template_embeddings)
    # Rank by score descending
    ranked_indices = np.argsort(scores)[::-1]
    results: List[Dict[str, Any]] = []
    for idx in ranked_indices[: max(1, limit)]:
        item = dict(templates[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results
