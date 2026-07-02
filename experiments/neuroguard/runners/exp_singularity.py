# -*- coding: utf-8 -*-
"""특이점 탐색: (1) 신종 공격 발굴 = 진화탐색으로 5개 공격레인(W/J/B/K/F) 혼합 genome을
코디네이터 곱셈이 '최소'가 되도록 진화 → 인간이 안 짠 공격 조합을 기계가 발굴.
(2) 미지 강건성(held-out) = 코디네이터는 공격을 모른 채 고정 휴리스틱이므로, 발굴된 신종에
코디네이터가 얼마나 버티나 = 미지 공격 강건성 지표.
누적: 매 실행 결과를 summary_singularity_log.csv 에 append (계속 돌리며 특이점 추적)."""
import sys, os, csv
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"
SEEDS_SEARCH = [3000, 3001, 3002]; SEEDS_EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10], "K": [3, 3], "F": [7, 8]}
JAM_VECS = {"J", "F"}; LANES = ["W", "J", "B", "K", "F"]; LMLEN = 20
KNOWN_WEAKEST = 0.726  # 코디네이터가 가장 약한 알려진 공격(재밍) 곱셈 — 이보다 낮으면 신종 강공격

def lanes_from_genome(g):
    g = np.asarray(g, float); g = g / g.sum()
    cnt = np.floor(g * LMLEN).astype(int)
    while cnt.sum() < LMLEN: cnt[np.argmax(g * LMLEN - cnt)] += 1
    lm = []
    for i, c in enumerate(cnt): lm += [LANES[i]] * int(c)
    return lm[:LMLEN] if lm else ["W"] * LMLEN

def make_genome_red(lanemap):
    class R(brains._Red):
        LM = list(lanemap)
        def get_action(self, obs, asp):
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5, obs)
            o = int(self.name.split("_")[-1]); lane = self.LM[o % len(self.LM)]
            return self._emit(int(self.np_random.choice(VEC_AIDS[lane])), obs)
    return R

def retake_target(env, a, node, ip2d, sleep):
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node: return i
    c = idx.get("RetakeControl", []); return c[0][0] if c else sleep

def rollout(seed, red, vectors, coord=True):
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]
    cf, af = [], []; k = len(vectors)
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        acts = {}; dejam = 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
        used = set(); assign = {}
        for c in comp:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        for a in live:
            i = int(a.split("_")[-1])
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx)
            elif i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else:
                if red_jam > 0: dejam += 1
                acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - red_jam + min(dejam, red_jam)) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av, final

def fitness(g, seeds):
    lm = lanes_from_genome(g); red = make_genome_red(lm)
    rs = [rollout(s, red, lm) for s in seeds]
    return float(np.mean([r[0] for r in rs])), float(np.mean([r[1] for r in rs]))

def evolve(gens=6, pop=10, rng=None):
    rng = rng or np.random.default_rng(0)
    # seed population: pure lanes + known combos + random
    P = [np.eye(5)[i] + 1e-3 for i in range(5)]
    P += [np.array([1, 1, 1, 0, 0.]), np.array([0, 1, 0, 0, 1.])]  # WJB, JF
    while len(P) < pop: P.append(rng.dirichlet(np.ones(5)))
    hist = []
    best = None
    for gen in range(gens):
        scored = []
        for g in P:
            mult, comp = fitness(g, SEEDS_SEARCH); scored.append((mult, comp, g))
        scored.sort(key=lambda x: x[0])  # lower 곱셈 = stronger attack
        if best is None or scored[0][0] < best[0]: best = scored[0]
        hist.append(scored[0][0])
        gtxt = ",".join(f"{c}:{v:.2f}" for c, v in zip(LANES, scored[0][2] / np.sum(scored[0][2])))
        print(f"  gen {gen+1}/{gens}  best곱셈 {scored[0][0]:.3f} (점령 {scored[0][1]:.3f})  [{gtxt}]", flush=True)
        # next gen: top-4 elites + mutations
        elites = [s[2] for s in scored[:4]]
        P = list(elites)
        while len(P) < pop:
            par = elites[rng.integers(len(elites))]
            child = par * np.exp(rng.normal(0, 0.6, 5))
            if rng.random() < 0.3: child[rng.integers(5)] += rng.random()
            P.append(np.abs(child) + 1e-3)
    return best, hist

if __name__ == "__main__":
    runseed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(f"=== 특이점 탐색: 신종 공격 발굴 (seed={runseed}, gens={gens}) ===", flush=True)
    best, hist = evolve(gens=gens, rng=np.random.default_rng(runseed))
    bmult, bcomp, bg = best; bg = np.asarray(bg) / np.sum(bg); lm = lanes_from_genome(bg)
    # held-out 재평가: full seeds
    red = make_genome_red(lm); rs = [rollout(s, red, lm) for s in SEEDS_EVAL]
    ho_mult = float(np.mean([r[0] for r in rs])); ho_comp = float(np.mean([r[1] for r in rs]))
    mix = ", ".join(f"{L}={w:.0%}" for L, w in zip(LANES, bg) if w > 0.05)
    novel = ho_mult < KNOWN_WEAKEST - 0.02
    print(f"\n발굴된 최강 신종 공격 혼합: {mix}")
    print(f"코디네이터 상대 곱셈 {ho_mult:.3f} (점령 {ho_comp:.3f})  | 알려진 최약 재밍 {KNOWN_WEAKEST}")
    print(f"판정: {'★특이점 후보 — 코디네이터가 모르던 더 강한 신종 발견' if novel else '코디네이터가 신종에도 버팀 = 미지 강건성 확인'}")
    logp = os.path.join(OUT, "summary_singularity_log.csv"); new = not os.path.exists(logp)
    with open(logp, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new: wr.writerow(["run", "best_mix", "곱셈", "점령", "known_weakest", "novel"])
        prev = sum(1 for _ in open(logp, encoding="utf-8")) - 1 if not new else 0
        wr.writerow([prev + 1, mix, round(ho_mult, 3), round(ho_comp, 3), KNOWN_WEAKEST, int(novel)])
    print(f"Saved/append summary_singularity_log.csv (run #{ (sum(1 for _ in open(logp, encoding='utf-8'))-1) })")
