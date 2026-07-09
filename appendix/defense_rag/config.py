"""defense_rag 공통 설정: 경로·모델명·상수 한 곳에 모음."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")            # 원본 덤프 (대용량, 미커밋 — download_data.sh)
RAG_DATA_DIR = os.path.join(HERE, "rag_data")    # 파생 산출물·임베딩 인덱스 (커밋됨, attack_rag/rag_data와 동일 컨벤션)

# 파싱 산출물 (build_kb.py)
TECHNIQUES_JSONL = os.path.join(RAG_DATA_DIR, "d3fend_techniques.jsonl")
ATTACK_MAP_JSON = os.path.join(RAG_DATA_DIR, "attack_to_d3fend.json")

# 벡터 인덱스 산출물 (index.py). 문서를 LangChain 청크로 쪼개 임베딩, numpy 행렬로 저장.
INDEX_NPZ = os.path.join(RAG_DATA_DIR, "d3fend_index.npz")

# 청킹 (langchain-text-splitters RecursiveCharacterTextSplitter, attack_rag와 동일 설정)
# chunk_size는 임베딩 모델(MiniLM, 256 wordpiece ≈ 약 1000자) 윈도에 맞춤 —
# 윈도 안 문서는 통짜 1청크(문맥 보존), 윈도 초과 문서만 분할(잘림 방지).
# attack_rag heldout 실측: 이보다 잘게(400자) 쪼개면 recall@5 0.533→0.479로 하락.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100     # 청크 경계에서 문맥 유실 방지

# 로컬 임베딩 모델 (오프라인, 384차원). 최초 1회만 다운로드 후 캐시.
EMBED_MODEL = "all-MiniLM-L6-v2"
