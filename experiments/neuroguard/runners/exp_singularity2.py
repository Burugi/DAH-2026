# -*- coding: utf-8 -*-
"""특이점 탐색 2차: 탐색공간에 '연결성 분할(frag_K)'까지 넣고, 최강 방어인 '통합 방어(ultimate:
per-segment+de-jam+relay)'를 상대로 진화. genome = (레인혼합 W/J/B, 분할강도 K). fitness = 통합방어
곱셈 '최소화'. 통합방어도 못 막는 조합이 나오면(곱셈 < 0.49) = 진짜 특이점, 아니면 = 통합방어가 천장 커버 증명.
누적: summary_singularity2_log.csv"""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"
SEEDS_S = [3000, 3001, 3002]; SEEDS_E = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}
LANES = ["W", "J", "B"]; LMLEN = 18; KMAX = 6
ULT_WORST = 0.509  # 알려진 통합방어 worst(A-CONN). 이보다 확실히 낮으면 특이점

def lanes_from_g(g):
    g = np.asarray(g, float); g = g / g.sum(); cnt = np.floor(g * LMLEN).astype(int)
    while cnt.sum() < LMLEN: cnt[np.argmax(g * LMLEN - cnt)] += 1
    lm = []
    for i, c in enumerate(cnt): lm += [LANES[i]] * int(c)
    return lm[:LMLEN] if lm else ["W"] * LMLEN

def make_red(lm):
    class R(brains._Red):
        LM = list(lm)
        def get_action(self, o, a):
            if self.mem.get("target") is not None and o.get("success") is True: return self._emit(5, o)
            i = int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.LM[i % len(self.LM)]])), o)
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

def retake_nearest(env, a, i, comp, pos, ip2d, sleep):
    if not comp: return sleep
    tgt = min(comp, key=lambda c: np.linalg.norm(pos[i] - pos[c]))
    idx = actions.action_index_map(env, a)
    for j, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == tgt: return j
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, red, mode, vectors, frag_K=0, R_relay=4):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:frag_K]) if frag_K else set()
    k = len(vectors); cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]; A = adjacency(pos, ml)
        present = set(range(n)) - hubs; comps = components(present, A, n)
        big = max(comps, key=len) if comps else set(); isolated = present - big
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS); jam = red_jam > 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        reconnected = set(list(isolated)[:R_relay]) if mode == "ultimate" else set()
        iso2 = isolated - reconnected
        seg_groups = comps if mode == "ultimate" else [big | reconnected]
        unreachable = hubs if mode == "ultimate" else (hubs | iso2)
        assign = {}
        for seg in seg_groups:
            sc = [c for c in seg if c in comp]; scl = [d for d in seg if d not in comp]; used = set()
            for c in sc:
                cand = sorted([d for d in scl if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
                if cand: assign[cand[0]] = c; used.add(cand[0])
        dejam = 0; acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign: acts[a] = retake_nearest(env, a, assign[i], comp, pos, ip2d, sleep)
            elif mode == "ultimate" and jam: dejam += 1; acts[a] = actions.make_blue_index(1, env, a, ctx)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        restored = min(dejam, red_jam) if mode == "ultimate" else 0
        af.append(max(0.0, (n - len(comp) - len(hubs) - len(iso2) - (red_jam - restored)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av

def fitness(gw, kf, seeds, mode="ultimate"):
    lm = lanes_from_g(gw); red = make_red(lm); K = int(round(kf))
    return float(np.mean([rollout(s, red, mode, lm, frag_K=K) for s in seeds]))

def evolve(gens=6, pop=9, rng=None):
    rng = rng or np.random.default_rng(0)
    P = [(np.eye(3)[i] + 1e-3, 0.0) for i in range(3)]
    P += [(np.array([1, 1, 1.]), float(k)) for k in (2, 4, 6)]  # 동시공격 + 분할
    while len(P) < pop: P.append((rng.dirichlet(np.ones(3)), rng.uniform(0, KMAX)))
    best = None
    for gen in range(gens):
        scored = sorted([(fitness(gw, kf, SEEDS_S), gw, kf) for gw, kf in P], key=lambda x: x[0])
        if best is None or scored[0][0] < best[0]: best = scored[0]
        gw = scored[0][1] / np.sum(scored[0][1])
        print(f"  gen {gen+1}/{gens}  통합방어곱셈 {scored[0][0]:.3f}  [W{gw[0]:.2f} J{gw[1]:.2f} B{gw[2]:.2f} | K={int(round(scored[0][2]))}]", flush=True)
        elites = [(s[1], s[2]) for s in scored[:4]]; P = list(elites)
        while len(P) < pop:
            pw, pk = elites[rng.integers(len(elites))]
            cw = np.abs(pw * np.exp(rng.normal(0, 0.6, 3))) + 1e-3
            ck = float(np.clip(pk + rng.normal(0, 1.5), 0, KMAX))
            P.append((cw, ck))
    return best

if __name__ == "__main__":
    runseed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(f"=== 특이점 2차: 연결성 포함, 통합방어 상대 (seed={runseed}, gens={gens}) ===", flush=True)
    best = evolve(gens=gens, rng=np.random.default_rng(runseed))
    ul, gw, kf = best; gw = np.asarray(gw) / np.sum(gw); lm = lanes_from_g(gw); K = int(round(kf))
    red = make_red(lm)
    ul_e = float(np.mean([rollout(s, red, "ultimate", lm, frag_K=K) for s in SEEDS_E]))
    co_e = float(np.mean([rollout(s, red, "coord", lm, frag_K=K) for s in SEEDS_E]))
    mix = ", ".join(f"{L}={w:.0%}" for L, w in zip(LANES, gw) if w > 0.05) + f", 분할K={K}"
    sing = ul_e < ULT_WORST - 0.02
    print(f"\n발굴 최강(통합방어 상대): {mix}")
    print(f"통합방어 곱셈 {ul_e:.3f} | 코디네이터 {co_e:.3f} | 통합방어 알려진 worst {ULT_WORST}")
    print(f"판정: {'★★진짜 특이점 — 통합방어도 못 막는 신종' if sing else '통합방어가 천장까지 커버 = 특이점 없음(강건성 증명)'}")
    logp = os.path.join(OUT, "summary_singularity2_log.csv"); new = not os.path.exists(logp)
    with open(logp, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new: wr.writerow(["run", "mix", "통합방어곱셈", "코디네이터곱셈", "ult_worst", "singularity"])
        prev = (sum(1 for _ in open(logp, encoding="utf-8")) - 1) if not new else 0
        wr.writerow([prev + 1, mix, round(ul_e, 3), round(co_e, 3), ULT_WORST, int(sing)])
    print("Saved summary_singularity2_log.csv")
