# NeuroGuard — 드론 군집 사이버 방어 (HVT+RAG)
#
# sim(CybORG, numpy 1.23 고정)과 RAG(sentence-transformers)가 의존성이 충돌하므로
# 한 이미지 안에 venv 두 개(/opt/sim, /opt/rag)로 분리한다. README의 검증된 설치
# 절차를 그대로 옮긴 것.
#
# 빌드:  docker build -t neuroguard .
# 실행:
#   # 방어 채점 (기본 모델 = HVT+RAG)
#   docker run --rm neuroguard /opt/sim/bin/python src/score.py --scenario A17 --recall 0.75 --fp 0.1 --seeds 5
#   # HVT vs HVT+RAG 비교
#   docker run --rm --entrypoint bash neuroguard -c "PATH=/opt/sim/bin:$PATH ./src/compare_hvt_rag.sh"
#   # RAG 파이프라인 데모 (관측 → 공격판단 → 방어추천)
#   docker run --rm -w /app/appendix neuroguard /opt/rag/bin/python -m attack_rag.integration_test
#   # (선택) LLM 근거 생성 활성화
#   docker run --rm -w /app/appendix -e ANTHROPIC_API_KEY neuroguard /opt/rag/bin/python -m attack_rag.integration_test

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── env① sim: CybORG(CAGE Challenge 3, 커밋 고정) + 팀 requirements ──────────
# numpy==1.23.5 핀 유지를 위해 CybORG는 --no-deps로 설치 (README 절차와 동일)
RUN python -m venv /opt/sim \
    && git clone https://github.com/cage-challenge/CybORG /opt/CybORG \
    && cd /opt/CybORG && git checkout 2742b5e0ce4330c9b14006b38acd3b5ebe00d6fd
COPY requirements.txt .
RUN /opt/sim/bin/pip install --no-cache-dir -e /opt/CybORG --no-deps \
    && /opt/sim/bin/pip install --no-cache-dir -r requirements.txt

# ── env② rag: 임베딩·청킹·LLM 훅 (sim과 완전 분리) ──────────────────────────
COPY appendix/defense_rag/requirements.txt rag-requirements.txt
RUN python -m venv /opt/rag \
    && /opt/rag/bin/pip install --no-cache-dir -r rag-requirements.txt anthropic

# 임베딩 모델 사전 다운로드 → 컨테이너 실행 시 네트워크 불필요(오프라인 시연 보장)
RUN /opt/rag/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV SDL_VIDEODRIVER=dummy \
    TOKENIZERS_PARALLELISM=false

CMD ["/opt/sim/bin/python", "src/score.py", "--scenario", "A17", "--recall", "0.75", "--fp", "0.1", "--seeds", "5"]
