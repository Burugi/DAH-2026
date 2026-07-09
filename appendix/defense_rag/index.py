"""A3: D3FEND 방어 기법 271개 → LangChain 청킹 → 벡터 인덱스 구축 + 유사도 검색.

빌드: 기법 문서를 LangChain RecursiveCharacterTextSplitter로 청크 분할 후
각 청크를 임베딩한다(짧은 정의는 1청크, 긴 정의는 2청크+). 코퍼스가 작아
Chroma/FAISS 같은 벡터 DB 서버 없이 정규화된 임베딩 행렬 하나(.npz)에
청크→기법 매핑(doc_idx)과 함께 저장하고 코사인 유사도(=내적)로 검색한다.
완전 오프라인·의존성 최소·시연 중 네트워크 불필요.

검색: 청크 단위로 유사도를 구한 뒤 기법 단위 max로 집계해 top-k 기법을 반환
(어느 청크든 질의와 강하게 맞으면 그 기법이 올라온다).

  build_index()  : jsonl 읽어 청킹+임베딩 → rag_data/d3fend_index.npz 저장 (1회 실행)
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


def _chunks(text, label):
    """LangChain 스플리터로 청크 분할. 각 청크 앞에 기법명을 붙여 문맥 유지."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    parts = splitter.split_text(text)
    # 2번째 청크부터는 기법명이 잘려나가므로 접두어로 복원
    return [p if i == 0 else f"{label}: {p}" for i, p in enumerate(parts)]


def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def build_index():
    docs = _load_docs()
    model = _embedder()
    texts, doc_idx = [], []
    for i, d in enumerate(docs):
        for c in _chunks(_doc_text(d), d["label"]):
            texts.append(c)
            doc_idx.append(i)
    emb = model.encode(texts, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    meta = np.array([json.dumps(d, ensure_ascii=False) for d in docs], dtype=object)
    np.savez(config.INDEX_NPZ, emb=emb, meta=meta,
             doc_idx=np.array(doc_idx, dtype=np.int32))
    print(f"[index] 기법 {len(docs)}개 → LangChain 청크 {len(texts)}개 임베딩 "
          f"-> {config.INDEX_NPZ}  (dim={emb.shape[1]})")
    return len(docs)


class Retriever:
    """벡터 검색 경로(②): 자유 텍스트 관측 -> 유사 D3FEND 기법 top-k (청크 max 집계)."""

    def __init__(self):
        z = np.load(config.INDEX_NPZ, allow_pickle=True)
        self.emb = z["emb"]                                  # (청크수, dim) 정규화됨
        self.docs = [json.loads(m) for m in z["meta"]]
        # 청크→기법 매핑. 구버전(청킹 전) npz면 1청크=1기법으로 간주
        self.doc_idx = (z["doc_idx"] if "doc_idx" in z.files
                        else np.arange(len(self.docs), dtype=np.int32))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _embedder()
        return self._model

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        chunk_scores = self.emb @ q                          # 청크 단위 코사인 유사도
        scores = np.full(len(self.docs), -1.0, dtype=np.float32)
        np.maximum.at(scores, self.doc_idx, chunk_scores)    # 기법 단위 max 집계
        idx = np.argsort(-scores)[:k]
        out = []
        for i in idx:
            d = dict(self.docs[i])
            d["score"] = round(float(scores[i]), 3)
            out.append(d)
        return out


if __name__ == "__main__":
    build_index()
