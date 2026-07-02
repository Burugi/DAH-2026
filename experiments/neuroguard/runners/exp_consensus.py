# -*- coding: utf-8 -*-
"""A3. Communication/CONSENSUS loop under satellite intermittency. Defender no longer gets the true
global compromise view for free; each connected drone observes locally + neighbors, gossips. Blacked-out
drones (fraction p) can't observe/act. Defense acts on the GOSSIPED believed set (1-step latency).
Compare perfect-view vs gossip-consensus across blackout p."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg=yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml",encoding="utf-8"))
OUT=r"C:\workspace\DAH2026_exp"; EVAL=[3000,3001,3002,3003,3004]; PS=[0.0,0.2,0.4,0.6]
def frontier(i,comp,pos,ml): return any(d!=i and np.linalg.norm(pos[i]-pos[d])<ml for d in comp)
def neighbors(i,pos,ml,n): return [j for j in range(n) if j!=i and np.linalg.norm(pos[i]-pos[j])<ml]
def pred_act(i,comp,pos,ml):
    if i in comp: return 3
    return 8 if (comp and frontier(i,comp,pos,ml)) else (4 if comp else 1)
def rollout(seed,p,mode):
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,brains.RuleRed); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    rng=np.random.default_rng(seed+555); k=int(round(p*n))
    black=set(int(x) for x in rng.choice(n,size=k,replace=False)) if k else set()
    believed=set()                       # gossiped shared view (1-step latency)
    cf=[]
    for t in range(cfg["steps"]):
        true_comp=run.compromised_drones(cyborg,n); cf.append(len(true_comp)/n)
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        if mode=="perfect":
            view=true_comp - black
        else:  # gossip consensus: each connected drone observes self+neighbors; union over connected; blacked-out invisible
            obs=set()
            for i in range(n):
                if i in black: continue
                if i in true_comp: obs.add(i)
                for j in neighbors(i,pos,ml,n):
                    if j in true_comp and j not in black: obs.add(j)
            view=set(believed) | obs       # 1-step latency: act on previous belief merged with new obs
            believed=obs                   # update belief for next step
        ctx={"compromised":view,"ip_to_drone":ip2d,"n":n}
        live=[a for a in env.active_agents if a in env.agent_actions]
        acts={}
        for a in live:
            i=int(a.split("_")[-1])
            aid=0 if i in black else pred_act(i,view,pos,ml)
            acts[a]=actions.make_blue_index(aid,env,a,ctx)
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    return len(run.compromised_drones(cyborg,n))/n
def ev(p,mode): return float(np.mean([rollout(s,p,mode) for s in EVAL]))
print("=== A3 consensus(gossip) vs perfect-view under satellite blackout p (rule웜, 점령) ===")
print("p".ljust(6)+"perfect-view   gossip-consensus")
rows=[]
for p in PS:
    pf=ev(p,"perfect"); gs=ev(p,"gossip"); rows.append((p,pf,gs))
    print(f"{p:<6.1f}{pf:14.3f}{gs:16.3f}")
with open(os.path.join(OUT,"summary_consensus.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["blackout_p","perfect","gossip"])
    for r in rows: wr.writerow([r[0],round(r[1],3),round(r[2],3)])
print(f"\n단절0%: perfect {rows[0][1]:.3f} vs gossip {rows[0][2]:.3f} (통신비용)")
print(f"단절60%: perfect {rows[-1][1]:.3f} vs gossip {rows[-1][2]:.3f} (합의 붕괴)")
print("Saved summary_consensus.csv")
