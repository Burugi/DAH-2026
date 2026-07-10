# 데모 & 시각화

## HVT 샘플 (A1 공급망 웜)

`docs/sample_run/`은 HVT 방어를 A1(공급망 웜) 시나리오에서 실행한 결과 번들이다. CybORG 설치 없이
아래 명령으로 대시보드와 애니메이션을 만들 수 있다.

```bash
python src/viz/dashboard.py docs/sample_run --png   # -> dashboard.html + dashboard_preview.png
python src/viz/render.py docs/sample_run --gif      # -> figs/animation.gif
```

방어점수만 직접 채점하려면 `score.py`를 쓴다(대시보드 번들은 만들지 않고 점수·step 상태만 출력).

```bash
python src/score.py --scenario A1                  # 방어 점수 (기본 모델 = HVT+RAG)
python src/score.py --scenario A1 --log steps.csv  # step별 상태 CSV
./src/compare_hvt_rag.sh                           # HVT vs HVT+RAG 비교 (데모 영상 수치 재현)
```

## 직접 실행해 기록하기 (HVT+RAG 데모)

`record.py`가 방어 모델 실행을 `results/<run_id>/` 번들로 저장해, 위 시각화 도구를
샘플이 아닌 **방금 돌린 실행**에 그대로 쓸 수 있다 (데모 영상 녹화용).

```bash
# 1) 실행 기록 (기본 모델 = rag-guided(HVT+RAG) · 옵션: --model hvt · reach2)
python src/record.py --scenario A17 --recall 0.75 --fp 0.1

# 2) 대시보드 HTML(인터랙티브) + 프리뷰 PNG + 애니메이션 GIF
python src/viz/dashboard.py results/rag_guided_A17_r0.75_fp0.1
python src/viz/dashboard.py results/rag_guided_A17_r0.75_fp0.1 --png
python src/viz/render.py   results/rag_guided_A17_r0.75_fp0.1 --gif

# 3) 브라우저로 열기 (재생/일시정지·스크럽)
open results/rag_guided_A17_r0.75_fp0.1/dashboard.html      # macOS
```

RAG 검색 파이프라인 자체(관측→RAG-A→RAG-B 추천)를 터미널로 보여주려면 (별도 임베딩 env):

```bash
cd appendix && python -m attack_rag.integration_test
```

**대시보드 프리뷰** — 한 화면에 함대 맵 + 공격/방어 점수 곡선 + 전술 로그.
웜이 퍼지지만 HVT가 RetakeSuspicious로 봉쇄해 방어점수 D≈0.94, 가용성 0.88을 유지한다.

![dashboard](sample_run/dashboard_preview.png)

**함대 애니메이션** — step별 재생.

![animation](sample_run/figs/animation.gif)

범례: 삼각형 = UAV, 네모 = UGV, 빨강 = 감염, 보라 링 = 재밍, 주황 화살표 = GPS 스푸핑(실제→보고),
노랑 링 = 탐지. `dashboard.html`은 재생/일시정지·스크럽이 되는 인터랙티브 버전이다.

## 데모 영상

- 링크: https://youtu.be/w4Xyg006Jx0
- 구성: ① 노벨 공격 식별(RAG-A→RAG-B) → ② 방어 성능 비교(HVT vs HVT+RAG) → ③ 실전 대시보드(A17)

## 행동 단독 검증

각 행동이 실제로 동작하고 효과가 다름을 확인한 예 (`src/gallery.py` 산출물).

![seize](gifs/action_red_SeizeControl.gif)
![retake](gifs/action_blue_RetakeSuspicious.gif)

## 시각화 명령

```bash
# 인터랙티브 뷰어 (SPACE 재생/정지, ←/→ step, R 리셋, ESC 종료)
python src/viz/render.py results/<run>

# step별 애니메이션 GIF   -> results/<run>/figs/animation.gif
python src/viz/render.py results/<run> --gif

# 정적 그림             -> results/<run>/figs/*.png
python src/viz/plot.py results/<run>

# 대시보드 HTML + 프리뷰 PNG
python src/viz/dashboard.py results/<run> --png
```

베이스라인 스윕(`sweep.py`)은 매 매치업마다 `dashboard.html`과 그림을 자동 생성하고, 상위에
`grid_heatmaps.png`(3×3 요약)와 `summary.csv`를 쓴다.

아래는 `render.py`로 만든 베이스라인 매치업 애니메이션 예시다.

![render 예시](example_animation.gif)
