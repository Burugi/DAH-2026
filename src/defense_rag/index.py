"""A3: D3FEND 방어 기법 271개 → 벡터 인덱스 구축 + 유사도 검색.

문서 271개는 아주 작은 코퍼스라 Chroma/FAISS 같은 벡터 DB 서버 없이
정규화된 임베딩 행렬 하나(.npz)에 저장하고 코사인 유사도(=내적)로 검색한다.
완전 오프라인·의존성 최소·시연 중 네트워크 불필요.

  build_index()  : jsonl 읽어 임베딩 → d3fend_index.npz 저장 (1회 실행)
  Retriever      : npz 로드 후 .search(질의문, k) → top-k 방어 기법
"""
import json

import numpy as np

from . import config


def _load_docs():
    docs = []
    with open(config.TECHNIQUES_JSONL) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def _doc_text(d):
    """임베딩에 넣을 검색용 텍스트. 이름·동의어·정의를 합쳐 질의 매칭력을 높인다."""
    syn = ", ".join(d.get("synonyms") or [])
    parts = [d["label"]]
    if syn:
        parts.append(f"(also: {syn})")
    parts.append(d["definition"])
    return " ".join(parts)


def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def build_index():
    docs = _load_docs()
    model = _embedder()
    texts = [_doc_text(d) for d in docs]
    emb = model.encode(texts, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    meta = np.array([json.dumps(d, ensure_ascii=False) for d in docs], dtype=object)
    np.savez(config.INDEX_NPZ, emb=emb, meta=meta)
    print(f"[index] {len(docs)}개 기법 임베딩 -> {config.INDEX_NPZ}  (dim={emb.shape[1]})")
    return len(docs)


class Retriever:
    """벡터 검색 경로(②): 자유 텍스트 관측 -> 유사 D3FEND 기법 top-k."""

    def __init__(self):
        z = np.load(config.INDEX_NPZ, allow_pickle=True)
        self.emb = z["emb"]                                  # (N, dim) 정규화됨
        self.docs = [json.loads(m) for m in z["meta"]]
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _embedder()
        return self._model

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        scores = self.emb @ q                                # 코사인 유사도
        idx = np.argsort(-scores)[:k]
        out = []
        for i in idx:
            d = dict(self.docs[i])
            d["score"] = round(float(scores[i]), 3)
            out.append(d)
        return out


if __name__ == "__main__":
    build_index()
