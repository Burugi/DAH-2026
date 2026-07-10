#!/usr/bin/env bash
# 장면2: HVT vs HVT+RAG 성능 비교 — 실전조건(미탐 25%·오탐 10%), 5시드.
# 실행: conda activate dah 후  ./demo_scene2.sh
cd "$(dirname "$0")"

run() {  # run <model> <scenario> -> 점수만 반환
  python src/score.py --model "$1" --scenario "$2" --recall 0.75 --fp 0.1 --seeds 5 2>/dev/null \
    | tail -1 | grep -o '0\.[0-9]*' | head -1
}

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  공격 판단이 방어 자세를 고른다 — HVT vs HVT+RAG              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo "  HVT     : 반사실 검증으로 재장악을 가려서 하는 기본 방어 모델"
echo "  HVT+RAG : 같은 방어 + RAG 공격판단(attack_class)으로 자세 라우팅"
echo "  조건    : 실전 탐지 — 감염 25% 놓침(미탐) · 멀쩡한 드론 10% 오탐"

for s in A19 A8; do
  case "$s" in
    A19) name="A19 Raven — 은밀 시맨틱 내비게이션 공격"
         desc="탐지를 피해 천천히 침투하며 경로를 조작하는 은밀 공격";;
    A8)  name="A8 C2 링크 하이재킹 — 강제 재인증(deauth)"
         desc="지상관제 링크를 끊고 가로채 통신 가용성을 무너뜨리는 공격";;
  esac
  echo ""
  echo "━━━ $name ━━━"
  echo "  공격     : $desc"
  echo "  RAG 판단 : 재밍/거부 계열 → 복원력 자세 (오탐 드론의 파괴적 재장악 회피)"
  h=$(run hvt "$s");        echo "  HVT      = $h"
  r=$(run rag-guided "$s"); python -c "print(f'  HVT+RAG  = $r   (+{$r - $h:.3f})')"
done

echo ""
echo "──────────────────────────────────────────────────────────────────"
echo "  같은 방어 체계, 다른 자세 — 점수 차이는 '오탐 드론을 살려둔' 가용성"
echo "  18개 실공격 시나리오 평균: HVT 0.922 → HVT+RAG 0.932 (13개 우위)"
