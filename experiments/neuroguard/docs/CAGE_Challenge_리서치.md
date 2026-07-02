> **DAH 2026 전략 문서(`DAH2026_전략문서.md`) §5의 근거 자료.** 2026-06-25 리서치. 모든 주요 주장은 1차 출처(GitHub 저장소, arXiv 논문, 공식 사이트)에 근거하며, 검증되지 않은 부분은 명시.

---

# CAGE Challenge 종합 리서치 보고서
### (Cyber Autonomy Gym for Experimentation) — 자율 사이버 방어 AI 경진대회/벤치마크 분석

## 0. 핵심 요약 (Executive Summary)

- **CAGE Challenge**는 TTCP(The Technical Cooperation Program, 5개국 국방과학 협력체)의 CAGE 워킹그룹이 주관하고, 호주 국방과학기술그룹(**DSTG**)이 주도하는 **자율 사이버 방어(Autonomous Cyber Operations, ACO) 에이전트 개발용 공개 경진대회 시리즈**.
- 모든 대회는 **CybORG(Cyber Operations Research Gym)** RL 시뮬레이션 환경 위에서 진행. 참가자는 주로 **blue agent(방어자)** 를 학습·제출하고, 정해진 **red agent(공격자)** 와 대결, **green agent(정상 사용자)** 가 노이즈 생성.
- **CC1(2021), CC2(2022), CC3(2022~23), CC4(2024)** 4개 iteration. 단일 호스트/네트워크 방어 → 기만 강화 → **드론 군집 분산 다중 에이전트** → **대규모 기업망 MARL** 로 발전.
- **상금 없음**(리더보드 순위 + 논문 발표가 인센티브).
- **핵심 발견: CC3·CC4 모두에서 잘 설계된 휴리스틱(규칙 기반) 에이전트가 딥러닝/MARL을 능가** → 한국형 국방 AI 해커톤 전략에 중요한 시사점.

> **브리프 정정(1차 출처 대조):** ① CybORG 시뮬레이터는 "behaviour tree"가 아니라 **유한상태기계(FSM)** 기반. ② CybORG++ 경량 엔진은 "Mephisto"가 아니라 **MiniCAGE**이며 **CAGE 2 기반**(CC4 아님). ③ **OpenC2/CASTLE** 구현 증거 없음(미확인). ④ 행동의 MITRE ATT&CK 매핑은 확인됨.

---

## 1. CAGE Challenge란 — 목적·주관·기원·목표

### 1.1 목적과 배경
- AI로 **머신 속도·규모의 분산·적응형 자율 사이버 방어**가 가능하다는 기대. CybORG와 CAGE 시나리오로 **ACO의 TTP** 개발 지원, 사이버 보안 문제를 **AI/ML 커뮤니티가 접근 가능한 형태로 공개**.
- 공개 배포는 **시뮬레이터에 한정**(에뮬레이터/실제 인프라는 비공개).
- (출처: github.com/cage-challenge ; dst.defence.gov.au CC3 공모)

### 1.2 주관 조직
- **TTCP**(호주·캐나다·뉴질랜드·영국·미국) 산하 **CAGE 워킹그룹**.
- **호주 DSTG**가 CybORG 개발·대회 주도(2021 정규논문 저자 6명 전원 DSTG).
- CC4 공저자: DSTG(호주)·Dstl(영국)·DRDC(캐나다)·NSA/NIWC Pacific/DEVCOM/NRL/AFRL(미국)·University of Canterbury(뉴질랜드) + 학계/산업(Cornell, Punch Cyber, Cybermonic).
- (출처: arxiv.org/abs/2108.09118 ; onlinelibrary.wiley.com/doi/full/10.1002/aaai.70021)

### 1.3 기원·역사
- **2020:** "CybORG: An Autonomous Cyber Operations Research Gym"(arXiv:2002.10667) — 초기엔 자율 침투테스트 도구.
- **2021.08:** 정규 논문(arXiv:2108.09118, DSTG) + CC1 동시 공개(IJCAI-21 ACD).
- 이후 **CC2(2022) → CC3(2022~23) → CC4(2024)**.

### 1.4 전체 목표
- RL 기반 **자율 사이버 방어(부분적으로 공격) 에이전트**를 개발·평가·벤치마킹하는 **공통 환경 + 표준 문제 + 공개 경진대회**.

---

## 2. 각 Iteration 상세

**공통 서사:** 가상국가 **Florin(아군) vs Guilder(적군)**. CC1/CC2는 군수공장 네트워크 방어, CC3/CC4는 드론 작전 배경.

### 2.1 CC1 — 단일 에이전트, 소규모 기업망 방어
- 공개 2021.08.20, Misinform 확장 2021.12.02, 마감 2022.02.07, 결과 2022.02.11(AAAI-22 AICS). 제출 7건, 순위 4팀.
- **네트워크:** 13호스트·3서브넷(User 5 / Enterprise 3+Defender / Operational: Op_Server0 + Op_Host 3). Subnet1→3 직접경로 없음(Subnet2 경유 pivot).
- **에이전트:** Red는 **고정/스크립트형**(B_line=최단직진 / Meander=점진장악 / Sleep). Green은 discovery만. 턴: blue→green→red.
- **Blue 행동 5종:** Monitor / Analyse / Remove / Restore(−1) / Misinform(디코이).
- **관측:** 52비트 비트벡터(호스트당 4비트), 탐지 노이즈 존재.
- **보상(전부 음수):** User admin −0.1/턴, Enterprise·Op_Server −1.0/턴, Op_Host −0.1/턴, red Impact −10, Restore −1.
- **평가:** 30/50/100스텝 × {B_line,Meander,Sleep}, 최종 1000에피소드 평균합(seed 117).
- (출처: github.com/cage-challenge/cage-challenge-1 ; arxiv.org/pdf/2310.13565)

### 2.2 CC2 — 기만(Deception) 중심화, 평가 강화
- 공개 2022.04.01, 마감 2022.07.18, 결과 2022.08.12. 리더보드 20개 항목.
- **네트워크:** CC1과 동일 13호스트·3서브넷. 핵심 변화는 **기만 확장 + 평가 강화**.
- **신규:** 디코이가 1급 행동으로 확장 — **7종 서비스 디코이**(DecoyApache/Femitter/HarakaSMPT/Smss/SSHD/Svchost/Tomcat). Red 익스플로잇 75% 상위/25% 임의. 평가 100에피소드(참가자), 최종 1000(seed 153). CybORG v2.1, 보상 벡터화.
- (출처: github.com/cage-challenge/cage-challenge-2 ; arxiv.org/abs/2309.07388 ; arxiv.org/abs/2211.15557)

### 2.3 CC3 — 드론 군집(Drone Swarm), 분산 다중 에이전트  🚁
> **드론 시나리오는 CC4가 아니라 CC3.**
- 공개 2022.09.30, 마감 2023.02.17, 결과 2023.03.03. 검증통과 3건(공개 "8팀 12접근법"). **우승 팀 Mindrake**(−1577.7).
- **시나리오:** Florin이 국경순찰용 **자율 드론 군집** 운용, **MANET**으로 통신 중계. 드론 펌웨어의 **하드웨어 트로이목마(공급망 악성코드)**가 활성화되면 **웜이 인접 드론으로 확산** → 병력이동 정보 탈취 또는 허위정보 주입.
- **환경(arXiv:2312.04940):** 드론 18대(임베디드 Linux+무선), 100×100 공간, 통신반경 30, 대역폭 100/스텝. 취약 서비스 포트 8888, 악성코드 매 스텝 ~5% 활성화. 위치 매 에피소드 무작위.
- **신규:** **드론당 blue 1개(총 18 blue) 분산 방어**, PettingZoo MARL 래퍼, MANET(토폴로지 매 스텝 변화), POMDP, **16비트 메시지** 브로드캐스트, 정찰단계 없음(즉시 공격).
- **보상:** 차단/도청/드롭/경로없음 각 −1, 완전장악 −(18×(t−500)). Sleep 평균 −8292.
- (출처: github.com/cage-challenge/cage-challenge-3 ; arxiv.org/abs/2312.04940)

### 2.4 CC4 — 대규모 기업망 MARL
- 공개 2024.02.20, 개발단계 종료 2024.03.29, 대회 종료 2024.05.10(결과 AAAI-25). **15팀/65제출**. 플랫폼 **Codalab**.
- **시나리오:** 드론 작전을 지원하는 대규모 군용 기업/운영망 방어 + **공식 MARL**.
- **네트워크:** 8서브넷. MISSIONNET(Operational A/B + Restricted A/B) + SIMNET(**Contractor Network**=인터넷 직결·blue 미보호·공격시작점, HQ Network). **호스트 수 에피소드마다 무작위**(서버 1~6, 사용자 3~10, 서비스 1~5).
- **신규:** **협력 MARL(Dec-POMDP)**, 에이전트 간 상호작용, 동적·확장 red, **임무 단계(Mission Phases)**(단계별 보상 가중치·통신정책 변화), 8비트 메시지.
- **Blue 행동 8종:** Monitor/Analyse/DeployDecoy/Remove/Restore/**BlockTrafficZone**/**AllowTrafficZone**/Sleep(강할수록 소요 스텝=페널티 큼).
- **Red:** `FiniteStateRedAgent`(FSM+확률행렬). **Green:** 모든 사용자 호스트, Local Work에 **PhishingEmail** 포함 가능(green이 red 유입).
- **보상 철학:** 임무 우선순위로 동적 스케일, **"blue/red가 green에 미치는 영향" 중심**(red 위치를 곧바로 모르게).
- (출처: github.com/cage-challenge/cage-challenge-4 ; cage-challenge.github.io/cage-challenge-4 ; 10.1002/aaai.70021)

---

## 3. CybORG 환경

- **이중성:** 빠른 **시뮬레이터** + 실제 인프라 **에뮬레이터**를 공통 인터페이스로 제공(시뮬 학습→실제 검증). 각 행동은 시뮬용 **상태전이** + 에뮬용 **실제 명령** 으로 이중 정의.
- **시뮬레이터 = 유한상태기계(FSM).** 에뮬레이터 = **AWS 실제 VM**(SSH+AWS CLI 배포, Metasploit/Meterpreter/Velociraptor 사용).
- **인터페이스:** 단일=OpenAI Gym(`reset`/`step`, 4-튜플), 다중=**PettingZoo Parallel**(CC3~). 래퍼: ChallengeWrapper, BlueTableWrapper, EnumActionWrapper, BlueEnterpriseWrapper(CC4) 등.
- **에이전트:** Blue(참가자 학습)/Red(스크립트형, **MITRE ATT&CK 매핑** 예: Discover→T1018, Exploit→T1210, Impact→T1489)/Green(양성 트래픽·오탐 생성=현실성 핵심).
- **관측:** 호스트명 키 중첩 딕셔너리 + success. RL용 평탄화(CC1 52차원). **보상:** 0에서 음수 페널티 누적, Impact 대량 감점, Restore도 감점(운영중단 비용), CC4는 **가용성(green 영향) 중심**.
- **상태모델:** State 클래스, link_diagram=NetworkX 그래프, blocks=방화벽/존 차단(에뮬=AWS Security Group/NACL).
- **CybORG++(arXiv:2410.16324):** Alan Turing Institute(영국 Dstl 지원), **CAGE 2 기반**, 경량 엔진 **MiniCAGE**(단일 CPU 수천 구성 병렬, ~1000× 속도, 보상상관 1.00, 버그 3종 수정).
- (출처: arxiv.org/abs/2108.09118 ; github.com/cage-challenge/CybORG ; arxiv.org/html/2410.16324v1)

---

## 4. 규칙·형식

- **제출:** CC1~3은 이메일 + 학습된 에이전트 코드. **CC4는 Codalab submission.zip 자동평가**. 산출물은 **보고서가 아니라 코드/모델**.
- **규칙(CC4):** 명명 규칙, 공정성(악용 시 리더보드 제거), 실행 제약(EC2 C4.large에서 100에피소드×500스텝 3시간 내).
- **평가:** 고정 red × 고정 길이 × 다수 에피소드 평균 누적보상(고정 시드, 95% CI). CC4는 100에피소드×500스텝 평균, 상위 4팀은 5종 변형 시나리오로 **일반화** 사후평가.
- **상금: 없음**(순위·논문·워크숍 발표·인지도).
- (출처: github.com/cage-challenge/cage-challenge-{1,2,4})

---

## 5. 경쟁자 기술 접근 — RL 알고리즘과 우승 방법

- **5.1 지배적: PPO.** CC1/CC2 상위 사실상 전부. DQN/DDQN은 부분관측·노이즈·희소보상에서 불안정해 폐기. 전형 64→64 ReLU MLP.
- **5.2 계층적 RL(대표 우승 패턴):** 상위 컨트롤러가 활동 중인 red 식별 → 전문 sub-policy 위임.
  - **Mindrake(Turing): CC1 우승**(Meander PPO + B_line curiosity-PPO + 밴딧 컨트롤러).
  - **Cardiff: CC2 우승**(Hierarchical PPO, 핑거프린팅 후 전문 정책 로드).
- **5.3 앙상블:** "Keeping it RL" CC2 2위(Ensemble of Ensembles, −56.90 vs 우승 −54.57).
- **5.4 기만(CC2 우승 레시피):** **PPO + 호스트별 탐욕적 디코이 배치**(9개 디코이 행동, 소진 시 Remove 폴백) + 스캔상태 패딩(메모리).
- **5.5 프레임워크:** 커스텀 PyTorch PPO, **Stable-Baselines3**(최다), RLlib(앙상블/분산), CleanRL. Gym/Gymnasium+PettingZoo.
- **5.6 보상 셰이핑(arXiv:2310.13565):** curiosity(ICM)는 광역 모니터링엔 별 도움 안 됨(좁은 하위과제엔 유효) → **셰이핑은 과제 의존적**.
- **5.7 🔑 CC3 우승 "Canaries and Whistles"(휴리스틱이 RL을 이김):** **Canaries**(정상 드론 ID 하트비트 → 멈추면 감염 추정) + **Whistles**(감염 ID 군집 재전파). **CW 휴리스틱 −1577.7 vs PPO −7617.8(약 5배 우수)**. 최선=혼합(7 PPO+11 CW = −1487.9).
- **5.8 🔑 CC4(휴리스틱이 모든 MARL을 이김):** Team UC 휴리스틱 −113(우승) / Lancer −118 / Punch −142 / **Cybermonic MARL(GNN+PPO) −193**. **Cybermonic 그래프 접근(arXiv:2509.16151):** 호스트=노드 GCN+PPO, 행동을 노드 단위 함수로 정의 → **네트워크 크기 자동 확장**(재학습 불필요), 미관측 토폴로지 zero-shot. **MARL 패배 이유:** 무작위 네트워크 적응 실패(무효 행동), 파라미터 공유 정보 병목.
- **5.9 교차 교훈:** ① 스크립트 red 과적합 위험(앙상블이 일반화 우수) ② **휴리스틱이 신규/복잡 환경에서 학습을 이김(CC3·CC4)** ③ **하이브리드(전문가+RL)가 실용 최전선** ④ 보상·관측 엔지니어링 > 알고리즘 선택 ⑤ MARL은 귀납적/그래프 표현 필요 ⑥ **Sim-to-real 격차(운영진도 경고: "Beyond games?")**.
- (출처: arxiv.org/abs/2211.15557 ; 2310.13565 ; 2312.04940 ; 2509.16151 ; github.com/john-cardiff/-cyborg-cage-2)

---

## 6. 호스팅 위치

- **GitHub(org `cage-challenge`):** CybORG / cage-challenge-1~4. 팀: john-cardiff/-cyborg-cage-2(CC2 우승), alan-turing-institute/cage-challenge-2-public(Mindrake), alan-turing-institute/CybORG_plus_plus.
- **사이트:** cage-challenge.github.io/cage-challenge-4 ; DSTG 공모(dst.defence.gov.au) ; Codalab(competitions/17672).
- **논문:** CybORG arXiv:2108.09118 / 2002.10667 ; CybORG++ 2410.16324 ; CC2 2309.07388·2211.15557 ; 보상셰이핑 2310.13565 ; CC3 우승 2312.04940 ; CC4 AI Magazine 10.1002/aaai.70021 ; Cybermonic 2509.16151.

---

## 7. CAGE vs 전형적 "공격/방어 AI 해커톤"

**유사:** red vs blue 구도, 국방/보안 도메인, 순위 경쟁, 보안 자동화 목표.

| 구분 | CAGE Challenge | 전형 해커톤(예: DAH) |
|---|---|---|
| 본질 | 시뮬 기반 **RL 벤치마크** | **인간 중심**(보고서/발표 심사 + 라이브 공방) |
| 제출물 | **학습된 자율 에이전트(코드/모델)** | 솔루션+보고서+발표, 또는 라이브 공방 |
| 평가 | 자동·정량(고정 시드, 수천 에피소드 평균) | 주관 심사(창의성·완성도) 또는 CTF |
| 상대 | 고정·스크립트 red(재현 가능, 비적응) | 실제 인간 상대팀 또는 동적 시나리오 |
| 기간 | 수개월 비동기 | 수일~수주 집중 |
| 현실성 | 추상 시뮬(sim-to-real 격차) | 실제 시스템에 가까움 |
| 상금 | **없음**(순위·논문) | **상금·채용 연계** 흔함 |
| 핵심 역량 | RL/MARL, 보상·관측 엔지니어링 | 도메인 전문성, 침투/방어 실전, 신속 프로토타이핑 |

**시사점:** CAGE는 "사람이 공방하는 행사"가 아니라 "AI 에이전트가 공방을 자동 수행하게 만드는 벤치마크". 재현 가능한 공통 환경의 장점 + 고정 스크립트 적군의 한계(과적합·비현실성) + 휴리스틱이 RL을 이긴 반복 결과(하이브리드/도메인 지식 장려가 현실적) + 상금형·논문형 결합의 가치.

---

## 검증 한계
- **상금:** 어떤 iteration도 금전 상금/공식 시상 언급 없음(미공개 시상은 확인 불가).
- **CC2 고유 팀 수, CC3 원래 일정(연장 전 1/31·2/14), DeepWiki 일부 수치(신뢰 낮음), CC4 서브넷 8 vs 9(Internet 포함 여부)** 등은 표기.
- **브리프 정정:** MiniCAGE(≠Mephisto), CybORG++=CAGE 2 기반, FSM(≠behaviour tree), Kali/pymetasploit3 논문 미명시, OpenC2/CASTLE 증거 없음.
- 일부 arXiv PDF 바이너리 반환 → 세부 점수 일부는 근사치(핵심 결과는 신뢰 가능).
