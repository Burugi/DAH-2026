# 데모 & 시각화

## HVT 샘플 (A1 공급망 웜)

`docs/sample_run/`은 HVT 방어를 A1(공급망 웜) 시나리오에서 3시드 실행한 결과다. CybORG 설치 없이
아래 명령으로 대시보드와 애니메이션을 재생성할 수 있다.

```bash
python src/viz/dashboard.py docs/sample_run --png   # -> dashboard.html + dashboard_preview.png
python src/viz/render.py docs/sample_run --gif      # -> figs/animation.gif
```

이 샘플은 다음으로 만들어졌다.

```bash
python src/run_hvt.py --scenario A1                 # -> results/hvt_A1/
python src/viz/dashboard.py results/hvt_A1 --png
python src/viz/render.py results/hvt_A1 --gif
```

**대시보드 프리뷰** — 한 화면에 함대 맵 + 공격/방어 점수 곡선 + 전술 로그.
웜이 퍼지지만 HVT가 RetakeSuspicious로 봉쇄해 방어점수 D≈0.94, 가용성 0.88을 유지한다.

![dashboard](sample_run/dashboard_preview.png)

**함대 애니메이션** — step별 재생.

![animation](sample_run/figs/animation.gif)

범례: 삼각형 = UAV, 네모 = UGV, 빨강 = 감염, 보라 링 = 재밍, 주황 화살표 = GPS 스푸핑(실제→보고),
노랑 링 = 탐지. `dashboard.html`은 재생/일시정지·스크럽이 되는 인터랙티브 버전이다.

## 데모 영상

- 링크: (녹화·발표 영상 링크를 여기에 추가)

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
