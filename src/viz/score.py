"""Composite attack / defense scores (numpy only, no CybORG so it runs offline).

Two role scores in [0,1], each a plain (unweighted) average of self-contained
sub-metrics, so a single number per side tells which agentic is best:

  attack_score  = mean(final_compromise, compromise_auc, 1 - ttf_norm, 1 - comp_F1)
  defense_score = mean(1 - final_compromise, 1 - compromise_auc, availability)
                  (comp_F1 제거: 감염 억제↑ → 탐지대상↓ → comp_F1↓ 역설로 좋은 방어를 감점)

availability = mean over steps and entities of (not compromised) AND (link up).
Per-step a_t / d_t drive the dashboard's running-score chart. Blue cumulative
reward and the extended detection F1s are reported separately (not folded in) to
keep the score self-contained and comparable across runs and scenarios.
"""
import numpy as np


def availability(red_owned, link_up):
    ro = np.asarray(red_owned).astype(bool)
    lu = np.asarray(link_up).astype(bool)
    return float(((~ro) & lu).mean())


def episode_scores(metrics, red_owned_all, link_up_all, steps):
    """metrics = the averaged per-matchup dict; *_all = stacked (seeds, steps, n)."""
    V = availability(red_owned_all, link_up_all)
    ttf_norm = min(1.0, metrics["time_to_first_compromise"] / max(1, steps))
    fc, cauc, cf1 = metrics["final_compromise"], metrics["compromise_auc"], metrics["comp_F1"]
    A = float(np.mean([fc, cauc, 1 - ttf_norm, 1 - cf1]))   # 공격측(진단): comp_F1 유지
    # ★방어점수에서 comp_F1 제거 — 역설: 감염을 잘 막을수록 탐지할 대상이 적어져 comp_F1↓
    #   → 좋은 방어가 오히려 감점(실측: reach2가 점령 0.022·가용성 0.886으로 최고인데
    #     comp_F1 0.403 때문에 열등 방어보다 낮게 순위). 가용성(V)은 유지 — 텔레메트리(재밍) 방어 반영.
    D = float(np.mean([1 - fc, 1 - cauc, V]))
    # 곱셈종합 = D_core × availability (대회식: 가용성=0이면 전체=0)
    D_core = float(np.mean([1 - fc, 1 - cauc]))
    D_mult = round(D_core * V, 3)      # = mean(1-fc, 1-auc) × 가용성
    return {"attack_score": round(A, 3), "defense_score": round(D, 3),
            "availability": round(V, 3), "D_mult": D_mult}


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
