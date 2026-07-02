# -*- coding: utf-8 -*-
"""Run ALL attack scenarios (A1~A21 + new) against the COORDINATOR by mapping each to its closest
implementable mechanism (existing primitive class and/or synthetic overlay). Reports 점령 | 가용성 |
곱셈종합 for each. A few (sidechannel/insider/exfil) have no compromise-spread model -> noted, not run.
Mechanisms: vectors(W/J/B) + overlays {inject, frag_K(연결성), blackout_p(위성), poison_q(Sybil/합의),
detector_q(적대적ML/VLM = 탐지 저하), tempo(은밀)}. Team metric comp_F1=0.866."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}


def make_red(vectors, tempo):
    class R(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, asp):
            if self.np_random.uniform() > tempo:
                return self._emit(0, obs)
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            o = int(self.name.split("_")[-1])
            return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[o % len(self.VECS)]])), obs)
    return R


def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1); return (d < ml) & (d > 0)


def largest_comp(present, A, n):
    seen = set(); best = set()
    for s in present:
        if s in seen: continue
        comp = set(); st = [s]
        while st:
            u = st.pop()
            if u in seen: continue
            seen.add(u); comp.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen: st.append(v)
        if len(comp) > len(best): best = comp
    return best


def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep


def rollout(seed, spec):
    vectors = spec.get("vectors", ["W"]); tempo = spec.get("tempo", 1.0)
    cfg2 = cfg
    if spec.get("start_red"):
        cfg2 = dict(cfg); cfg2["sim"] = dict(cfg["sim"]); cfg2["sim"]["starting_num_red"] = spec["start_red"]
    fleet, cyborg, env, ip2d = run.build_env(cfg2, seed, make_red(vectors, tempo)); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 4)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:spec.get("frag_K", 0)]) if spec.get("frag_K") else set()
    kbl = int(round(spec.get("blackout_p", 0.0) * n))
    black = set(int(x) for x in rng.choice(n, size=kbl, replace=False)) if kbl else set()
    k = len(vectors); cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        # connectivity fragmentation (frag) + satellite blackout -> unreachable
        present = set(range(n)) - hubs - black
        big = largest_comp(present, adjacency(pos, ml), n) if present else set()
        unreachable = hubs | black | (present - big)
        # detector degradation (adversarial ML / VLM) + Sybil/consensus poison -> what coordinator SEES
        seen = set()
        for i in comp:
            if i in unreachable: continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                seen.add(i)
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        # inject overlay: compromised drones drain a neighbour's availability
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in unreachable]
        assign = {}; used = set()
        for c in seen:                                   # coordinator assigns to DETECTED compromised
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue   # self-clean always
            if i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - len(unreachable) - red_jam - min(inj, max(0, n - len(comp))) ) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    # exfil scenarios: damage = undetected dwell (compromise AUC), not spread -> report AUC as 'exfil피해'
    return final, av, float(np.mean([1 - final, 1 - auc, COMP_F1])) * av, auc


def ev(spec):
    rs = [rollout(s, spec) for s in EVAL]
    return tuple(float(np.mean([r[j] for r in rs])) for j in range(4))


# scenario -> mechanism mapping
# 시나리오 = configs/attack_scenarios.yaml 단일 진실에서 로드 (exp_matrix와 동일 소스)
_SCEN_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "attack_scenarios.yaml")
SCEN = [(f"{s['id']} {s['name']}", {k: v for k, v in s.items() if k not in ("id", "name", "class")})
        for s in yaml.safe_load(open(_SCEN_YAML, encoding="utf-8"))["scenarios"]]

print("=== ALL scenarios vs COORDINATOR (점령 | 가용성 | 곱셈종합 | exfilAUC) — comp_F1=0.866 ===")
rows = []
for name, spec in SCEN:
    fc, av, m, auc = ev(spec); rows.append((name, fc, av, m, auc))
    extra = f" | 잠복AUC {auc:.3f}" if name.startswith("A16") else ""
    print(f"  {name:24} 점령 {fc:.3f} | 가용성 {av:.3f} | 곱셈 {m:.3f}{extra}")
rows_sorted = sorted(rows, key=lambda r: r[3])
print(f"\n가장 위협적(곱셈 최저): {rows_sorted[0][0]} {rows_sorted[0][3]:.3f}")
print(f"가장 약한 공격(곱셈 최고): {rows_sorted[-1][0]} {rows_sorted[-1][3]:.3f}")
print("주: A16 탈취는 확산 안 함 → 곱셈(점령기반)은 높게 보이나 '잠복AUC'가 실제 절취 피해 대용.")
with open(os.path.join(OUT, "summary_allscenarios.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["scenario", "점령", "가용성", "곱셈", "잠복AUC"])
    for r in rows: wr.writerow([r[0], round(r[1], 3), round(r[2], 3), round(r[3], 3), round(r[4], 3)])
print("DONE_ALL")
