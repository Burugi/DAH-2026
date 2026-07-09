#!/usr/bin/env bash
# A1: D3FEND 공식 덤프 다운로드 (크롤링 아님, MITRE 공식 배포 파일).
# 이 두 파일은 용량이 커서 git에 넣지 않는다 -> 이 스크립트로 재생성.
# 실행 후 `python -m defense_rag.build_kb` 로 파싱한다.
set -e
cd "$(dirname "$0")"

echo "[1/2] D3FEND 전체 온톨로지 (d3fend.json, ~4.7MB)"
curl -fSL "https://d3fend.mitre.org/ontologies/d3fend.json" -o d3fend.json

echo "[2/2] ATT&CK<->D3FEND 전체 추론 매핑 (~45MB)"
curl -fSL "https://d3fend.mitre.org/api/ontology/inference/d3fend-full-mappings.json" \
     -o d3fend-full-mappings.json

echo "완료. 다음: (src/에서) python -m defense_rag.build_kb"
