"""defense_rag 공통 설정: 경로·모델명·상수 한 곳에 모음."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# 파싱 산출물 (build_kb.py)
TECHNIQUES_JSONL = os.path.join(HERE, "d3fend_techniques.jsonl")
ATTACK_MAP_JSON = os.path.join(HERE, "attack_to_d3fend.json")

# 벡터 인덱스 산출물 (index.py). 271개 문서라 Chroma 없이 numpy 행렬로 저장.
INDEX_NPZ = os.path.join(HERE, "d3fend_index.npz")

# 로컬 임베딩 모델 (오프라인, 384차원). 최초 1회만 다운로드 후 캐시.
EMBED_MODEL = "all-MiniLM-L6-v2"
