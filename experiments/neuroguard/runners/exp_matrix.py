# -*- coding: utf-8 -*-
"""전 시나리오 × 방어 매트릭스: 23개 공격 시나리오(오버레이 포함)를 무방어/코디네이터/통합방어(ultimate)로
같은 seed에서 실측 비교. 방어별 곱셈종합 + 평균/worst 요약. (탐지저하 detector_q/poison_q 반영, 통합방어는
per-segment 분산자율+de-jam+relay)."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}

def make_red(vectors, tempo):
    class R(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, asp):
            if self.np_random.uniform() > tempo: return self._emit(0, obs)
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5, obs)
            o = int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[o % len(self.VECS)]])), obs)
    return R

def adjacency(pos, ml):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1); return (d < ml) & (d > 0)

def components(present, A, n):
    seen = set(); out = []
    for s in present:
        if s in seen: continue
        cc = set(); st = [s]
        while st:
            u = st.pop()
            if u in seen: continue
            seen.add(u); cc.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen: st.append(v)
        out.append(cc)
    return out

def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, spec, mode, R_relay=4, recall=1.0, fp=0.0):
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
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs - black
        comps = components(present, A, n); big = max(comps, key=len) if comps else set()
        isolated = present - big
        reconnected = set(list(isolated)[:R_relay]) if mode == "ultimate" else set()
        iso2 = isolated - reconnected
        if mode == "ultimate":
            seg_groups = comps; unreachable = hubs | black
        else:
            seg_groups = [big | reconnected]; unreachable = hubs | black | iso2
        # detection: 탐지된 감염만 코디네이터가 배정 (detector_q 저하·poison_q 오염)
        detected = set()
        for i in comp:
            if i in unreachable: continue
            if rng.random() < spec.get("detector_q", 1.0) and rng.random() >= spec.get("poison_q", 0.0):
                detected.add(i)
        # 현실 탐지: recall(놓침 FN) + fp(오탐). 방어는 believed로 결정, 채점은 true comp
        realistic = (recall < 1.0 or fp > 0.0)
        if realistic:
            detected = {i for i in detected if rng.random() < recall}
            fp_set = {i for i in range(n) if i not in comp and i not in unreachable and rng.random() < fp}
            believed = detected | fp_set
            heal_set = believed; target_set = believed; nonclean = believed
        else:
            heal_set = comp; target_set = detected; nonclean = comp
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        inj = len(comp) if spec.get("inject") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        acts = {}; dejam = 0
        if mode == "none":
            for a in live: acts[a] = actions.make_blue_index(1, env, a, ctx)   # 전원 관측(무방어)
        else:
            assign = {}
            for seg in seg_groups:
                sc = [c for c in seg if c in target_set]; scl = [d for d in seg if d not in nonclean]; used = set()
                for c in sc:
                    cand = sorted([d for d in scl if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                    if cand: assign[cand[0]] = c; used.add(cand[0])
            for a in live:
                i = int(a.split("_")[-1])
                if i in unreachable: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
                if i in heal_set: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
                if i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
                else:
                    if mode == "ultimate" and red_jam > 0: dejam += 1
                    acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam) if mode == "ultimate" else 0
        lost = len(unreachable) if mode != "ultimate" else (len(hubs) + len(black) + len(iso2))
        af.append(max(0.0, (n - len(comp) - lost - (red_jam - restored) - min(inj, max(0, n - len(comp)))) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

def ev(spec, mode, recall=1.0, fp=0.0): return float(np.mean([rollout(s, spec, mode, recall=recall, fp=fp) for s in EVAL]))

# 시나리오 = configs/attack_scenarios.yaml 단일 진실에서 로드 (원본 레포 src/configs 컨벤션)
_SCEN_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "attack_scenarios.yaml")
_SCEN_RAW = yaml.safe_load(open(_SCEN_YAML, encoding="utf-8"))["scenarios"]
SCEN = [(f"{s['id']} {s['name']}", {k: v for k, v in s.items() if k not in ("id", "name", "class")}) for s in _SCEN_RAW]
MODES = [("무방어", "none", 1.0, 0.0), ("코디네이터", "coord", 1.0, 0.0), ("통합방어", "ultimate", 1.0, 0.0),
         ("통합방어현실(r.75/fp.1)", "ultimate", 0.75, 0.1)]
print("=== 전 시나리오 × 방어 매트릭스 (곱셈종합, 같은 seed) ===", flush=True)
print("시나리오".ljust(20) + "".join(m.rjust(20) for m, *_ in MODES))
rows = []
for name, spec in SCEN:
    vals = [ev(spec, code, rec, fp) for _, code, rec, fp in MODES]
    rows.append((name, *vals))
    print(name.ljust(20) + "".join(f"{v:>20.3f}" for v in vals), flush=True)
print("-" * 90)
for j, (m, *_) in enumerate(MODES):
    col = [r[j + 1] for r in rows]
    print(f"{m:22} 평균 {np.mean(col):.3f}  worst {min(col):.3f}")
with open(os.path.join(OUT, "summary_matrix.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["scenario", *[m for m, *_ in MODES]]); wr.writerows([[r[0], *[round(x, 3) for x in r[1:]]] for r in rows])
print("DONE_MATRIX")
