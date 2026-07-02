# -*- coding: utf-8 -*-
"""Run ALL defense scenarios (B1~B20) by mapping each to a concrete mechanism on top of the
coordinator, measured against its TARGET attack. Baseline coordinator uses imperfect detection
(detect_q=0.85) so detection-defenses show value. Prevention-class defenses are modelled as
'exploit_fail' (reduce attacker success); response-class as detection/availability/inject/poison
modifiers. Reports 곱셈종합 (baseline vs +defense). Team metric comp_F1=0.866."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
sys.setrecursionlimit(10000)
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; EVAL = [3000, 3001, 3002, 3003, 3004]; COMP_F1 = 0.866
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}; JAM_VECS = {"J", "B"}


def make_red(vectors, tempo, exploit_fail):
    class R(brains._Red):
        VECS = list(vectors)
        def get_action(self, obs, asp):
            if self.np_random.uniform() > tempo: return self._emit(0, obs)
            if exploit_fail and self.np_random.uniform() < exploit_fail: return self._emit(0, obs)  # 예방형이 막음
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5, obs)
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


def rollout(seed, atk, dfn):
    vectors = atk.get("vectors", ["W"])
    red = make_red(vectors, atk.get("tempo", 1.0), dfn.get("exploit_fail", 0.0))
    fleet, cyborg, env, ip2d = run.build_env(cfg, seed, red); n = fleet["n"]; ml = cfg["fleet"].get("max_link", 40)
    rng = np.random.default_rng(seed + 5)
    pos0 = fleet["pos_true"][0]; deg = adjacency(pos0, ml).sum(1)
    hubs = set(int(x) for x in np.argsort(-deg)[:atk.get("frag_K", 0)]) if atk.get("frag_K") else set()
    detect_q = dfn.get("detect_q", 0.85)
    k = len(vectors); cf, af = [], []
    for t in range(cfg["steps"]):
        comp = run.compromised_drones(cyborg, n); cf.append(len(comp) / n)
        pos = fleet["pos_true"][min(t, fleet["steps"] - 1)]
        present = set(range(n)) - hubs
        big = largest_comp(present, adjacency(pos, ml), n) if present else set()
        isolated = present - big
        # distributed (B13/분산자율): isolated still self-defend (per-segment); else unreachable
        unreachable = hubs if dfn.get("distributed") else (hubs | isolated)
        # detection (B1/B5/B14) + poison (Sybil A10); poison_resist(B9) negates poison
        poison = 0.0 if dfn.get("poison_resist") else atk.get("poison_q", 0.0)
        seen = set()
        for i in comp:
            if i in unreachable: continue
            if rng.random() < detect_q and rng.random() >= poison: seen.add(i)
        red_jam = sum(1 for i in comp if vectors[i % k] in JAM_VECS)
        # inject damage (A2/A19); inject_block(B3/B17) negates it
        inj = (0 if dfn.get("inject_block") else (len(comp) if atk.get("inject") else 0))
        # dejam (B2/B12/B18): restore availability vs jamming
        restored_jam = red_jam if dfn.get("dejam") else 0
        ctx = {"compromised": comp, "ip_to_drone": ip2d, "n": n}
        live = [a for a in env.active_agents if a in env.agent_actions]
        sleep = actions.action_index_map(env, live[0]).get("Sleep", [(0, None)])[0][0] if live else 0
        clean = [int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in unreachable]
        assign = {}; used = set()
        for c in seen:
            cand = sorted([d for d in clean if d not in used], key=lambda d: np.linalg.norm(pos[d] - pos[c]))
            if cand: assign[cand[0]] = c; used.add(cand[0])
        acts = {}
        for a in live:
            i = int(a.split("_")[-1])
            if i in unreachable: acts[a] = actions.make_blue_index(0, env, a, ctx); continue
            if i in comp: acts[a] = actions.make_blue_index(3, env, a, ctx); continue
            if i in assign: acts[a] = retake_target(env, a, assign[i], ip2d, sleep)
            else: acts[a] = actions.make_blue_index(1, env, a, ctx)
        af.append(max(0.0, (n - len(comp) - len(unreachable) - (red_jam - restored_jam) - min(inj, max(0, n - len(comp)))) / n))
        _, rew, done, _ = env.step(acts)
        if all(done.values()): break
    final = cf[-1]; auc = float(np.mean(cf)); av = float(np.mean(af))
    return float(np.mean([1 - final, 1 - auc, COMP_F1])) * av


def ev(atk, dfn): return float(np.mean([rollout(s, atk, dfn) for s in EVAL]))


# (B, 설명, 클래스, target attack, defense modifier)
DEF = [
    ("B1 이상탐지", "탐지↑", {"vectors": ["W"]}, {"detect_q": 1.0}),
    ("B2 GPS탐지(RAIM)", "가용성복원", {"vectors": ["J"]}, {"dejam": True}),
    ("B3 MAVLink 인증", "주입차단(예방)", {"vectors": ["W"], "inject": True}, {"inject_block": True}),
    ("B4 망분할+IDS", "확산억제(예방)", {"vectors": ["W"]}, {"exploit_fail": 0.4}),
    ("B5 Canary&Whistle", "조기탐지", {"vectors": ["W"]}, {"detect_q": 1.0}),
    ("B6 기만(디코이)", "확산흡수", {"vectors": ["W"]}, {"exploit_fail": 0.5}),
    ("B7 가용성보존", "차단최소", {"vectors": ["W", "J", "B"]}, {"dejam": True}),
    ("B8 시큐어부트/증명", "초기침투차단(예방)", {"vectors": ["W"]}, {"exploit_fail": 0.3}),
    ("B9 비잔틴내성", "오염저항", {"vectors": ["W"], "poison_q": 0.5}, {"poison_resist": True}),
    ("B10 MTD(무빙타겟)", "표적교란(예방)", {"vectors": ["W"]}, {"exploit_fail": 0.3}),
    ("B11 제로트러스트", "세그먼트(예방)", {"vectors": ["W"]}, {"exploit_fail": 0.4}),
    ("B12 RF핑거프린팅", "재머식별→복원", {"vectors": ["J"]}, {"dejam": True}),
    ("B13 연합학습/분산", "분산자율", {"vectors": ["W", "J", "B"], "frag_K": 4}, {"distributed": True}),
    ("B14 적대적견고성", "탐지견고", {"vectors": ["W"]}, {"detect_q": 1.0}),
    ("B18 DoS방어", "가용성복원", {"vectors": ["J"]}, {"dejam": True}),
    ("B19 차등복구", "가용성인지", {"vectors": ["W", "J", "B"]}, {"dejam": True}),
]
NOTE = ["B15 디지털트윈 = 우리 MPC(곱셈 0.779, 별도 실측)",
        "B16 포렌식/IR = 사후(에피소드 후) → 실시간 점령확산 무관",
        "B17 PKI = 예방(키관리), B3와 동형(주입차단)",
        "B20 미지공격 4계층 = 적응 메타방어(별도 실측, 감지만으론 부족)"]

print("=== ALL defenses (B1~B20) — 코디네이터 base(detect_q=0.85) vs +방어 (곱셈종합) ===")
rows = []
for name, mech, atk, dfn in DEF:
    base = ev(atk, {})           # coordinator only (imperfect detection, no modifier)
    plus = ev(atk, dfn)          # + this defense
    rows.append((name, mech, base, plus, plus - base))
    arrow = "↑" if plus > base + 0.01 else ("=" if abs(plus - base) <= 0.01 else "↓")
    print(f"  {name:18}[{mech:10}] base {base:.3f} -> +방어 {plus:.3f}  {arrow}{plus-base:+.3f}")
print("\n--- 별도 실측/사후 ---")
for nt in NOTE: print("  " + nt)
with open(os.path.join(OUT, "summary_alldefenses.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f); wr.writerow(["defense", "mechanism", "base곱셈", "방어곱셈", "개선"])
    for r in rows: wr.writerow([r[0], r[1], round(r[2], 3), round(r[3], 3), round(r[4], 3)])
print("DONE_DEF")
