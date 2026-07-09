import json, pickle, numpy as np, sys, time
MODEL=sys.argv[1] if len(sys.argv)>1 else "sentence-transformers/all-MiniLM-L6-v2"
from sentence_transformers import SentenceTransformer
t0=time.time(); print(f"[load] {MODEL}", flush=True)
m=SentenceTransformer(MODEL)
kb=json.load(open('rag_data/attack_capec_kb.json',encoding='utf-8'))
ids=list(kb.keys())
corpus=[f"{kb[i]['name']}. {kb[i]['description']} {kb[i].get('detection','')}" for i in ids]
print(f"[embed KB] {len(ids)}개", flush=True)
E=m.encode(corpus,normalize_embeddings=True,batch_size=64,show_progress_bar=False)
# held-out (Enterprise 기법 대상만 — KB에 있음)
tests=json.load(open('rag_data/heldout_procedure_test.json',encoding='utf-8'))
import random; random.seed(0); tests=random.sample(tests,2000)  # 샘플 2000
print(f"[embed test] {len(tests)}개", flush=True)
Q=m.encode([t['text'] for t in tests],normalize_embeddings=True,batch_size=64)
sims=Q@E.T
order=np.argsort(-sims,axis=1)
gold=[t['gold'] for t in tests]
def rk(i):
    for r,j in enumerate(order[i][:50]):
        if kb[ids[j]]['id']==gold[i]: return r+1
    return 999
ranks=np.array([rk(i) for i in range(len(tests))])
print(f"\n[{MODEL}] held-out (n={len(tests)}, KB={len(ids)} incl ICS/Mobile/CAPEC)")
for k in [1,5,10,20]: print(f"  recall@{k}: {(ranks<=k).mean():.3f}")
print(f"  MRR: {(1/ranks[ranks<999]).mean():.3f}  | 시간 {time.time()-t0:.0f}s")
print(f"\n※ TF-IDF baseline: recall@5=0.294, recall@1=0.124")
