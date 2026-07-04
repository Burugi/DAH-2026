"""Defense-only performance score (numpy only, no CybORG so it runs offline).

최종 성능 지표는 순수 방어(예방) 성능이다. 공격측 점수·가용성·탐지F1은 방어 점수에서
제외한다:
  · 가용성 — 재밍 등 공격 효과가 혼재 (단일 시나리오 방어 측정이 목적)
  · comp_F1(탐지) — 감염이 0이면 탐지할 참positive가 없어 F1=0이 되어, 사전 차단으로
    감염을 원천봉쇄한 '방어 성공'을 오히려 감점시키는 역설. 예방과 구조적으로 충돌하므로 제외.

  defense_score = mean(1 - final_compromise, 1 - compromise_auc)   ← 보고용 최종 지표

두 항 모두 순수 예방: 최종 미점령 비율, 에피소드 평균 미점령 비율. 완벽 방어 → 1.0.
attack_score / availability / comp_F1 은 진단용으로만 함께 반환(보고 지표엔 미포함).
Per-step a_t / d_t 는 대시보드 차트용(별도).
"""
import numpy as np


def availability(red_owned, link_up):
    ro = np.asarray(red_owned).astype(bool)
    lu = np.asarray(link_up).astype(bool)
    return float(((~ro) & lu).mean())


def episode_scores(metrics, red_owned_all, link_up_all, steps):
    """metrics = the averaged per-matchup dict; *_all = stacked (seeds, steps, n).

    defense_score 가 보고용 최종 방어(예방) 성능(가용성·공격점수·탐지F1 제외).
    """
    V = availability(red_owned_all, link_up_all)               # 진단용
    ttf_norm = min(1.0, metrics["time_to_first_compromise"] / max(1, steps))
    fc, cauc, cf1 = metrics["final_compromise"], metrics["compromise_auc"], metrics["comp_F1"]
    A = float(np.mean([fc, cauc, 1 - ttf_norm, 1 - cf1]))      # 진단용(공격측)
    D = float(np.mean([1 - fc, 1 - cauc]))                     # ★ 방어 단일 성능(예방)
    # comp_F1 은 attack_defense_metrics 에서 이미 별도 컬럼으로 제공(진단용)
    return {"attack_score": round(A, 3), "defense_score": round(D, 3),
            "availability": round(V, 3)}


def per_step(red_owned, link_up):
    """Per-step attacker/defender score curves for one episode (steps, n) -> (steps,)."""
    ro = np.asarray(red_owned).astype(bool)
    lu = np.asarray(link_up).astype(bool)
    c = ro.mean(1)                              # compromised fraction per step
    v = ((~ro) & lu).mean(1)                    # availability per step
    a_t = (c + (1.0 - v)) / 2.0
    d_t = ((1.0 - c) + v) / 2.0
    return a_t.astype(float), d_t.astype(float)


def leaderboard(rows):
    """rows: list of dicts with keys red_type, blue_type, attack_score, defense_score.
    Returns (attackers, defenders) each a list of (agentic, mean_score) sorted desc."""
    reds, blues = {}, {}
    for r in rows:
        reds.setdefault(r["red_type"], []).append(r["attack_score"])
        blues.setdefault(r["blue_type"], []).append(r["defense_score"])
    atk = sorted(((k, round(float(np.mean(v)), 3)) for k, v in reds.items()),
                 key=lambda x: -x[1])
    dfn = sorted(((k, round(float(np.mean(v)), 3)) for k, v in blues.items()),
                 key=lambda x: -x[1])
    return atk, dfn
