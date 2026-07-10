#!/usr/bin/env bash
# 장면2: HVT vs HVT+RAG 성능 비교 — 실전조건(미탐 25%·오탐 10%), 5시드.
# 실행: conda activate dah 후  ./demo_scene2.sh
cd "$(dirname "$0")"

run() {  # run <model> <scenario> -> 점수만 반환
  python src/score.py --model "$1" --scenario "$2" --recall 0.75 --fp 0.1 --seeds 5 2>/dev/null \
    | tail -1 | grep -o '0\.[0-9]*' | head -1
}

for s in A19 A8; do
  name=$(python src/score.py --model hvt --scenario "$s" --seeds 0 2>/dev/null | head -1 \
         | sed 's/.*scenario=//; s/  seeds.*//')
  echo ""
  echo "━━━ $name  (실전조건: 미탐 25%·오탐 10%) ━━━"
  h=$(run hvt "$s");        echo "  HVT      = $h"
  r=$(run rag-guided "$s"); echo "  HVT+RAG  = $r"
  python -c "print(f'  성능차   = +{$r - $h:.3f}  ← RAG 공격판단이 방어 자세를 전환')"
done
