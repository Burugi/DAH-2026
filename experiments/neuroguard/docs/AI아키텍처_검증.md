# AI 에이전트 아키텍처 — 최신 연구 검증·보강 (2023~2025)

> **DAH 2026 ③ 아키텍처 검증 리서치 결과** (백그라운드 에이전트, 1차 출처 기반). CAGE/CybORG 너머 최신 동향.
> 관련: `시나리오_아이디에이션.md` PART C · `CAGE_Challenge_리서치.md`
> ⚠️ 동반 검증 3건(공격/방어/공급망)은 세션 한도로 **미완** → 리셋 후 재실행 예정.

## 핵심 결론 (먼저 읽기)
1. **우리 3-에이전트(Blue/Red/Green) 정식화 = 최신 SOTA와 정확히 일치.** CC4가 동일 구조(blue/red/green=가용성 기준)를 Dec-POMDP로 정식화, green 미작동 시 blue 패널티. → "발명"이 아니라 **검증된 표준의 채택**(강점).
2. **"가용성 인식 보상"도 이미 표준.** `λ·가용성손실` = CC4 green-penalty + 동적 CIA 가중 보상과 동일. 채점식 ×가용성을 보상엔 **덧셈 패널티**로 넣은 게 학습 안정성 측면에서 오히려 옳음.
3. **하이브리드 코어(휴리스틱+PPO+LLM+가드)는 방향은 옳으나 15일엔 과함.** 최신 정석은 "LLM이 RL을 **대체**"가 아니라 "**LLM이 RL을 부트스트랩/보강**". 권장: 휴리스틱 baseline + PPO(action-masking) + **LLM은 보상설계·계획·설명에 오프라인 활용**. 추론루프 투입은 데모용 한정.

## 1. LLM 사이버보안 에이전트
- **LLM=ACD (2025, arXiv:2505.04843):** CC4에서 LLM 방어자 vs RL 직접 비교. LLM은 최소학습·신규공격 적응 우수하나 **장기 교전 누적보상은 RL 우위**. → 우리 과제와 가장 직접 일치.
- **DRL+증강LLM 하이브리드 (Computer Networks 2025, S1389128625001306):** LLM-informed proactive DRL이 휴리스틱 red 대비 impact action을 더 크게 지연. (전문 유료)
- **Red(침투) 실효성 냉정:** 완전자율 end-to-end 성공률 **~31%**(arXiv:2512.14233); 반면 PentestGPT v2+Opus 4.1은 13머신 중 10개(76.9%), Excalibur CTF 최대 91%(arXiv:2602.17622). **자율 LLM red는 신뢰성 낮음·무한루프** → 해커톤 Red는 **FSM+ATT&CK 스크립트형**이 안정·재현적.
- PentestGPT: github.com/GreyDGL/PentestGPT

## 2. 휴리스틱+RL 하이브리드 / 보상 / Safe RL
- **LLM+RL 통합 레시피 (arXiv:2509.05311):** PPO + **inference-only action masking + teacher-loss(σ 감쇠)**. 초기보상 2배, ~4,500ep 빠른 수렴. **단일행동 추천은 정책 99.96% 과첨예화 → LLM 출력을 "확률분포"로 줘야 안정.** 8B=16GiB·3.69초/스텝 → 실시간 부적합.
- **보상 셰이핑:** Eureka식 LLM 보상설계(arXiv:2511.16483); 환경설계 전문가 합의(arXiv:2604.08805) **4원칙: 목표 정렬 / 과도한 셰이핑·휴리스틱 직접 인코딩 회피 / 희소 보상 선호 / reward hacking 방지.**
- **Safe RL (arXiv:2505.17342):** CMDP+Lagrangian / **Shielding·Action Masking**(경량, 사전 차단) / Lyapunov. SMT 제약 RL(arXiv:2104.08994). → 15일엔 **hard action-masking**으로 충분.

### 우리 보상식 평결
- `R_blue = Σ위협차단 − Σ침해손실 − λ·가용성손실` (덧셈): ✅ **타당**(CC4 구조 일치).
- `R_red = Σ공격목표 × availability_factor` (곱셈): ⚠️ **정정 필요** — reward hacking·그래디언트 소실/폭주. → **가중합으로 분해**: `R_red = Σ공격목표 − μ·노출위험`. **가용성 곱셈은 "채점식"에만, 보상은 덧셈으로.**
- 셰이핑 과다 주의 → **희소·최소 셰이핑**, potential-based shaping만.

## 3. 가용성/임무 인식 보상 선행
- **CC4가 최강 선행:** green이 임무구역에서 일 못하면 큰 패널티 = "가용성+임무단계". (cage-challenge.github.io/cage-challenge-4)
- 동적 CIA 가중(재학습 없이 A 우선), HVT 격리·복구 회피 학습. (arXiv:2306.09318, 2404.10788) ; 임무인식 복원력(dl.acm.org/10.1145/3579375.3579421).
- **개선:** 위성망 "임무 구간(전송 윈도우)"을 phase로 모델링 → ×가용성과 정렬.

## 4. CPS/드론에 RL/LLM
- **UGV/군용차량 사이버 IR-RL (arXiv:2410.21407):** 우리 UGV에 직결, 직접 인용처.
- 드론망 DRL(2312.04940), UAV swarm anti-jam MARL(IEEE 10107729, arXiv:2512.16813), FANET EMARL(Nature s41598-026-39366-x), UAV IDS 서베이.
- **시사점:** 대부분 "통신 복원력"이지 "호스트 침해 사이버 공방"이 아님 → **네트워크/사이버 + 물리/통신 두 층 구분**이 차별점. 임베디드 제약 → **TinyRL/Decision Transformer**(2402.13201) 경량 정책.

## 5. sim-to-real
- 환경설계 합의(2604.08805): **virtualisation gap + modelling gap** 분해. 처방: **관측을 실제 센서/로그에 grounding(magic state 금지) + 지연·노이즈 domain randomization + LLM은 추론루프에서 제거**.
- **teacher-student distill:** LLM teacher(시뮬) → 경량 RL student → 임베디드 배포. (CyGIL 2508.19278, 2304.01244)

## 6. 15일·백지 팀 권장 아키텍처
| 계층 | 구성 | 근거 |
|---|---|---|
| 환경 | **CC4/Cyberwheel/CybORG++ 채택**, 직접구현 금지 | 2604.08805 |
| 안전가드(먼저) | **hard action-masking**(green 차단 금지, 임무 중 핵심호스트 격리·재시작 제한) | 2505.17342, 2104.08994 |
| Blue 코어 | 휴리스틱 baseline → **PPO+action-masking** → (여유)bandit/앙상블 | CybORG 검증 |
| Red | **FSM+ATT&CK 스크립트** + (여유)self-play RL | 자율 LLM red 31% |
| Green | CC4 내장 | 가용성=채점 직결 |
| LLM(오프라인) | 보상설계 + PPO 부트스트랩(action prior **확률분포**) + incident 설명 | 추론루프 배제(지연) |
| 학습 | 커리큘럼 + 제한적 self-play | CurriculumPT 등 |
| 평가 | 보상 외 **MTTD/MTTR·가용성유지율·침해호스트수** + 다중시드 | 2604.08805 |

**타임박스:** D1–3 환경+휴리스틱+가드 / D4–9 PPO(masking·희소보상·λ가용성) / D10–12 LLM 오프라인+제한 self-play / D13–15 지표·데모.
**하지 말 것:** LLM 매스텝 추론 / Red·Blue 둘 다 LLM자율 / SMT·Lyapunov / 환경 직접구현 / 곱셈·과다셰이핑 보상.

## 7. 최종 평결
| 요소 | 평결 | 정정 |
|---|---|---|
| Blue/Red/Green POMDP | ✅ SOTA 정합(강점) | CC4 채택 |
| 하이브리드 코어 | 🟡 방향 옳음, 범위 과함 | LLM→오프라인, 휴리스틱=masking, 가드=hard shield |
| R_blue(덧셈+λ) | ✅ 타당 | 셰이핑 최소·희소화 |
| R_red(곱셈) | ⚠️ 정정 | 가중합 분해 |
| self-play/커리큘럼 | ✅ 정합 | 커리큘럼 필수, self-play 제한 |
| sim-to-real | ✅ 인식 옳음 | grounding+randomization+teacher-student |

## 주요 출처
arXiv: 2505.04843, 2509.05311(✅전문), 2604.08805(✅전문), 2505.17342, 2104.08994, 2310.13565, 2602.04809, 2602.17622, 2512.14233, 2511.16483, 2410.21407, 2312.04940, 2306.09318, 2404.10788, 2508.19278, 2304.01244, 2402.13201, 2410.16324, 2501.14700, 2509.16151 · CC4: cage-challenge.github.io/cage-challenge-4 · PentestGPT: github.com/GreyDGL/PentestGPT · awesome-rl-for-cybersecurity(Kim-Hammar)
*한계: 일부 ScienceDirect/arXiv 전문 추출 실패(초록·스니펫 기반, 핵심 주장 교차확인). 2026 발행분은 프리프린트 가능성.*
