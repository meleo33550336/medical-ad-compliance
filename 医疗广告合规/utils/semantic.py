from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


_model = None


def load_model(name='paraphrase-multilingual-MiniLM-L12-v2'):
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def semantic_matches(text, examples, model=None, top_k=3, threshold=0.65):
    """计算文本与敏感示例的相似度，返回超过阈值的匹配项。"""
    if model is None:
        model = load_model()
    texts = [text] + examples
    embs = model.encode(texts, convert_to_numpy=True)
    q = embs[0:1]
    pool = embs[1:]
    sims = cosine_similarity(q, pool)[0]
    results = []
    for i, s in enumerate(sims):
        if s >= threshold:
            results.append({'example': examples[i], 'score': float(s)})
    # 排序高分优先并返回 top_k
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    return results
