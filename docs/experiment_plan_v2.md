# 실험 계획 v2 — 다양한 Blue 방어 아키텍처 × 시나리오 상보성

> 목적: 챔피언 1개를 고르는 게 아니라, **아키텍처마다 잘 막는 공격이 다르다**는 것을
> 정량으로 보여 보고서를 풍부하게 만든다. 산출물의 핵심은 **아키텍처 × 위협클래스
> D_mult 매트릭스**(누가 어디서 이기는가) + 페어드 유의성 검정.

## 0. 환경 상태 (복구 완료)
- CybORG 3.1(CC3 DroneSwarm) 재클론·재설치 완료 (`/Users/sumin/Desktop/DAH/CybORG`, `dah` conda env).
- 스모크 테스트 통과: `rule=0.554, hier_h3_tight=0.593, ooda=0.568` (A1, 2시드) — 기존 문서와 정합.
- `ANTHROPIC_API_KEY` **미설정** → 모든 llm/hier commander는 결정론적 오프라인 stub로 동작.

## 1. 평가 프로토콜 (고정)
- **측정**: 방어 성능만. 지표 = **정본 `src/viz/score.py` 정의** 사용:
  `D_mult = mean(1-final_comp, 1-comp_auc) × availability` — **comp_F1 제외**.
  - comp_F1은 방어점수에서 의도적으로 빠져 있음(score.py 주석): 감염을 잘 막을수록 탐지대상이
    줄어 comp_F1↓ → 좋은 방어가 감점되는 역설 때문.
  - ⚠️ `analyze_significance.dmult_of` / `run_stress.py`는 comp_F1을 **포함**한 다른(legacy) 정의.
    이전 실험 결과는 불필요하므로 두 파일은 **수정하지 않고 폐기**하고, 신규는 정본만 사용.
  - **단일 출처**: `score.py`에 per-seed 헬퍼 `d_mult_single(red_owned, link_up, n)`를 추가하고
    신규 분석은 전부 이를 호출(공식이 score.py 한 곳에만 존재). `dmult_of`는 import 안 함.
  - 부가로 availability(V), final_comp, comp_auc, defense_score를 분해 기록.
- **공격(red)**: rule / llm / rl 고정. 1차는 **rule-red**(가장 강함, 기존 baseline 정합), 2차 robustness로 3종 평균.
- **방어(blue)**: base.py(`BlueBrainBase`) 상속 신규 아키텍처 + 비교 기준선(rule, ooda, hier_h3_tight).
- **시나리오**: 23종 전부, **시나리오 단위**로 개별 측정(집계 안 함) → 클래스별 강점 추출.
  - 위협클래스(기존 매핑): 연결성(A_CONN,A_MV) · 가용성(A4,A7,A8,A13) · 탐지회피(A3,A6,A11,A19) · 점령(나머지 13종).
- **시드**: 1차 10시드(속도), 최종 유의성용 15시드. 페어드 설계((시나리오,시드) 쌍 고정, blue만 교체).
- **통계**: 기존 `analyze_significance` 함수 재사용 — 페어드 t / Wilcoxon / 부트스트랩 CI / Cohen's d_z / 승률.

## 2. 신규 Blue 아키텍처 후보 (창의적 구성)
모두 `BlueBrainBase` 상속, `BLUE_MULTIAGENT_TYPES`에 등록. 성능 동력이 per-drone dispatch임이
ablation으로 확인됐으므로 대부분 `team_decide`(드론별 aid) 방식으로 구현.

| # | 이름 | 핵심 루프/기술 | 대응 예시(사용자 제시) | 예상 강점 클래스 |
|---|------|--------------|----------------------|----------------|
| A | **RAGPlaybook** | 위협 시그니처 → 사전 플레이북 KB 검색(RAG) 후 stance/정책 적용 | "정찰단계 사전 RAG" | 탐지회피·가용성(시그니처 뚜렷) |
| B | **BanditFeedback** | 스탠스 대상 컨텍스추얼 밴딧, step reward 피드백으로 온라인 학습(RLHF-lite) | "피드백 루프 강화학습" | 점령·연결성(장기 적응) |
| C | **EnsembleMoE** | 전문가 3종(rule/ooda/hier) 드론별 가중투표+게이팅 | "하이브리드 코어" | 전 클래스 제너럴리스트(저분산) |
| D | **Predictive** | SIR형 확산 예측 → 다음 감염후보 선제 차단/탈환 | "최신 기술: 예측 방어" | 점령·확산(웜/캐스케이드) |
| E | **GraphCentrality** | 링크 그래프 중심성(degree/근사 betweenness)으로 허브 우선 보호 | "GNN 대체 그래프 인지" | 연결성·가용성 |
| F | **Debate** (선택) | 공격/보존 두 페르소나 토론 → 심판 조정(실 Claude 쇼케이스) | "에이전틱 루프 강화" | 서사·정성 보강 |

각 아키텍처는 오프라인 stub로 완전 동작(무키·재현), 키 있으면 동일 인터페이스로 실제 Claude 호출.

## 3. 실행 단계
- **Phase 1 (오프라인 전체 매트릭스)**: 신규 5종 + 기준선 3종 × 23시나리오 × 10시드 × rule-red ≈ 1,840 롤아웃.
  산출: `results/arch_matrix/pairs.csv`, 클래스별 D_mult 히트맵, 시나리오별 승자표.
- **Phase 2 (유의성)**: 상위 후보 15시드로 재측정 + 페어드 검정 → `docs/arch_matrix_validation.md`.
- **Phase 3 (선택, 실 Claude)**: 키 제공 시 대표 6~8 시나리오만 실제 Claude로 재실행, stub 대비 갭 측정.

## 4. 산출물 형식
- CSV: (scenario, seed, threat_class, [각 blue의 D_mult]) — 스프레드시트 이식용.
- 그림: 아키텍처×클래스 D_mult 히트맵 + 시나리오별 최고 아키텍처 막대.
- 표: 클래스별 승자 + 페어드 검정(p, CI, 승률).
- 마크다운 리포트 1종(기존 significance 문서 스타일).

## 5. 확정 필요 사항
1. Claude 호출 모드: 오프라인 stub 전체(즉시/무료/재현) vs 키 제공 후 실 Claude 서브셋.
2. 구축할 아키텍처 세트: 5종 전부 vs 우선 서브셋.
3. red 범위: rule-red 1차만 vs 3종(rule/llm/rl) 평균까지.
