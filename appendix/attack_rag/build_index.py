# rag_data/attack_index.npz 재생성 — KB 1456개를 LangChain 청킹 후 임베딩 (1회 실행).
# 실행: appendix/ 에서 `python -m attack_rag.build_index`
# 저장 키: E(청크 임베딩) · xwE(크로스워크 임베딩, 신호가 짧아 청킹 불필요) · doc_idx(청크→KB 매핑)
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attack_rag.rag_a import kb_chunks, _DATA, _DEFAULT_MODEL
from sentence_transformers import SentenceTransformer


def main():
    kb = json.load(open(os.path.join(_DATA, 'attack_capec_kb.json'), encoding='utf-8'))
    xw = json.load(open(os.path.join(_DATA, 'drone_crosswalk.json'), encoding='utf-8'))
    ids = list(kb.keys())
    m = SentenceTransformer(_DEFAULT_MODEL)
    texts, doc_idx = kb_chunks(kb, ids)
    E = m.encode(texts, normalize_embeddings=True, batch_size=64,
                 show_progress_bar=False).astype(np.float32)
    xwE = m.encode([x['itext'] for x in xw],
                   normalize_embeddings=True).astype(np.float32)
    out = os.path.join(_DATA, 'attack_index.npz')
    np.savez(out, E=E, xwE=xwE, doc_idx=doc_idx)
    print(f"[index] KB {len(ids)}개 → LangChain 청크 {len(texts)}개 임베딩 -> {out}  (dim={E.shape[1]})")


if __name__ == "__main__":
    main()
