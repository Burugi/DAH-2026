# -*- coding: utf-8 -*-
"""[신규 D1] 탐지 무결성 방어 실측. 공격: 지속형 스텔스(일부 감염 드론이 '지속적으로' 저탐지)·FP 오염(클린을
감염으로 오탐). 방어: naive(노드당 단일 관측) vs quorum(이웃 m대 상호증명 다수결). 채점=true comp.
가설: quorum이 독립 노이즈(FP·정상 FN)는 크게 잡지만, '지속 스텔스'(상관 회피)는 남는다 = 잔여 위협 규명."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}

def make_red(v):
    class R(brains._Red):
        VECS = list(v)
        def get_action(self, o, a):
            if self.mem.get("target") is not None and o.get("success") is True: return self._emit(5, o)
            i = int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[i % len(self.VECS)]])), o)
    return R

def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def neighbors(pos, i, m):
    d = np.linalg.norm(pos - pos[i], axis=1); return list(np.argsort(d)[1:m + 1])

def rollout(seed, red, vectors, mode, stealth_frac=0.0, stealth_recall=0.2, base_recall=0.85, fp=0.0):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]
    stealth = set(np.random.default_rng(seed).choice(n, size=int(stealth_frac * n), replace=False)) if stealth_frac else set()
    m_obs = 5 if mode == "quorum" else 1
    cf, af = [], []; k = len(vectors); S = {}
    for t in range(cfg["steps"]):
        true_comp = set(run.compromised_drones(cyborg, n)); cf.append(len(true_comp) / n)
        rng = np.random.default_rng(seed * 1000 + t); pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        # 방어의 believed: naive(단일)·quorum(다수결)·cusum(시간누적 순차탐지)
        believed = set()
        for tgt in range(n):
            rec = (stealth_recall if tgt in stealth else base_recall) if tgt in true_comp else fp
            if mode == "cusum":
                flagged = rng.random() < rec
                S[tgt] = max(0.0, S.get(tgt, 0.0) + (1.0 if flagged else -0.5))
                if S[tgt] >= 2.0: believed.add(tgt)
            else:
                votes = int(np.sum(rng.random(m_obs) < rec))
                if votes >= (m_obs // 2 + 1): believed.add(tgt)
        red_jam = sum(1 for i in true_comp if vectors[i % k] in JAM_VECS)
        ctx = {"compromised": believed, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in believed]
        used = set(); assign = {}
        for c in believed:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        acts = {}; dejam = 0
        for a in live:
            i = int(a.split("_")[-1])
            if i in believed: acts[a] = actions.make_blue_index(3, env, a, ctx)
            elif i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                if red_jam > 0: dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(true_comp) - red_jam + min(dejam, red_jam)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

CONDS = [("기준(스텔스0/fp0)", dict()), ("지속스텔스40%", dict(stealth_frac=0.4)),
         ("FP오염 fp0.25", dict(fp=0.25)), ("스텔스40%+FP0.25", dict(stealth_frac=0.4, fp=0.25))]
red = make_red(["W"])
print("=== [D1] 탐지 무결성: naive vs quorum(이웃5) vs cusum(시간누적), 곱셈종합 ===", flush=True)
print("공격 조건".ljust(22) + "naive".rjust(10) + "quorum".rjust(10) + "cusum".rjust(10))
rows = []
for label, kw in CONDS:
    na = float(np.mean([rollout(s, red, ["W"], "naive", **kw) for s in EVAL]))
    qu = float(np.mean([rollout(s, red, ["W"], "quorum", **kw) for s in EVAL]))
    cu = float(np.mean([rollout(s, red, ["W"], "cusum", **kw) for s in EVAL]))
    rows.append((label, round(na, 3), round(qu, 3), round(cu, 3)))
    print(f"{label:22}{na:>10.3f}{qu:>10.3f}{cu:>10.3f}", flush=True)
with open(os.path.join(OUT, "summary_detintegrity.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["condition", "naive", "quorum", "cusum"]); wr.writerows(rows)
print("DONE_DETINT")
