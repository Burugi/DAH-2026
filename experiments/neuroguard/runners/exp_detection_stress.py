# -*- coding: utf-8 -*-
"""[비판 검증] 코디네이터는 comp(감염집합)를 ground-truth로 읽어 결정에 쓴다 = 탐지 오라클 가정.
현실은 탐지가 불완전(재현율 recall<1=놓침 FN, 오탐 fp>0=클린을 감염으로 착각). 방어는 '믿음(believed)'으로
결정하고, 채점은 '진실(true)'로 한다. recall·fp 스윕으로 오라클 가정이 성능을 얼마나 부풀렸는지 실측."""
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

def rollout(seed, red, vectors, recall, fp):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]
    cf, af = [], []; k = len(vectors)
    for t in range(cfg["steps"]):
        true_comp = set(run.compromised_drones(cyborg, n)); cf.append(len(true_comp) / n)   # 채점=진실
        rng = np.random.default_rng(seed * 1000 + t)
        # 방어의 믿음: 진감염은 recall 확률로 탐지(FN=1-recall), 클린은 fp 확률로 오탐
        believed = {c for c in true_comp if rng.random() < recall}
        believed |= {i for i in range(n) if i not in true_comp and rng.random() < fp}
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
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
            if i in believed: acts[a] = actions.make_blue_index(3, env, a, ctx)      # 믿음상 감염 → 자가치료
            elif i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)  # 믿음상 감염 노드 재장악
            else:
                if red_jam > 0: dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(true_comp) - red_jam + min(dejam, red_jam)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av, final

ATTACKS = [("웜", ["W"]), ("동시(W+J+B)", ["W", "J", "B"])]
GRID = [("오라클(recall1.0 fp0)", 1.0, 0.0), ("recall0.9", 0.9, 0.0), ("recall0.75", 0.75, 0.0),
        ("recall0.5", 0.5, 0.0), ("recall0.75+fp0.15", 0.75, 0.15), ("recall0.5+fp0.2", 0.5, 0.2)]
print("=== [검증] 탐지 오라클 가정이 코디네이터를 얼마나 부풀렸나 (곱셈종합) ===", flush=True)
hdr = "탐지품질".ljust(22) + "".join(f"{a:>14}" for a, _ in ATTACKS)
print(hdr)
rows = []
for label, rec, fp in GRID:
    line = label.ljust(22); vals = []
    for an, vec in ATTACKS:
        red = make_red(vec)
        m = float(np.mean([rollout(s, red, vec, rec, fp)[0] for s in EVAL]))
        vals.append(round(m, 3)); line += f"{m:>14.3f}"
    print(line, flush=True); rows.append([label, *vals])
with open(os.path.join(OUT, "summary_detection_stress.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["detection", *[a for a, _ in ATTACKS]]); wr.writerows(rows)
print("DONE_DETSTRESS")
