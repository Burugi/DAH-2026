# -*- coding: utf-8 -*-
"""Consensus-POISONING attack on the A3 gossip loop: compromised drones inject false "I'm clean"
telemetry, so a fraction q of compromised drones are EXCLUDED from the believed set -> not retaken
-> fester. Sweep poison rate q. Defense = coordinator on the (poisoned) believed set."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg=yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml",encoding="utf-8"))
OUT=r"C:\workspace\DAH2026_exp"; EVAL=[3000,3001,3002,3003,3004]; COMP_F1=0.866
def retake_target(env,a,node,ip2d,sleep):
    idx=actions.action_index_map(env,a)
    for i,ip in idx.get("RetakeControl",[]):
        if ip2d.get(ip)==node: return i
    c=idx.get("RetakeControl",[]); return c[0][0] if c else sleep
def rollout(seed,q):
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,brains.RuleRed); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    rng=np.random.default_rng(seed+808); cf,af=[],[]
    for t in range(cfg["steps"]):
        comp=run.compromised_drones(cyborg,n); cf.append(len(comp)/n)
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        # poisoning: each compromised drone hides ('I'm clean') with prob q -> excluded from belief
        believed=set(i for i in comp if rng.random()>=q)
        ctx={"compromised":believed,"ip_to_drone":ip2d,"n":n}
        live=[a for a in env.active_agents if a in env.agent_actions]
        sleep=actions.action_index_map(env,live[0]).get("Sleep",[(0,None)])[0][0] if live else 0
        clean=[int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
        assign={}; used=set()
        for c in believed:                       # coordinator assigns only to BELIEVED compromised
            cand=sorted([d for d in clean if d not in used],key=lambda d:np.linalg.norm(pos[d]-pos[c]))
            if cand: assign[cand[0]]=c; used.add(cand[0])
        acts={}
        for a in live:
            i=int(a.split("_")[-1])
            if i in comp: acts[a]=actions.make_blue_index(3,env,a,ctx); continue   # self-clean still works locally
            if i in assign: acts[a]=retake_target(env,a,assign[i],ip2d,sleep)
            else: acts[a]=actions.make_blue_index(1,env,a,ctx)
        af.append(max(0.0,(n-len(comp))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=cf[-1]; auc=float(np.mean(cf)); av=float(np.mean(af))
    return final,float(np.mean([1-final,1-auc,COMP_F1]))*av
def ev(q): rs=[rollout(s,q) for s in EVAL]; return tuple(float(np.mean([r[j] for r in rs])) for j in range(2))
print("=== 합의 오염(Sybil/허위 텔레메트리) 공격 — 코디네이터 (rule웜, 점령|곱셈) ===")
print("poison q".ljust(10)+"점령      곱셈")
rows=[]
for q in [0.0,0.2,0.4,0.6,0.8]:
    fc,m=ev(q); rows.append((q,fc,m)); print(f"{q:<10.1f}{fc:.3f}    {m:.3f}")
with open(os.path.join(OUT,"summary_consensus_attack.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["poison_q","점령","곱셈"])
    for r in rows: wr.writerow([r[0],round(r[1],3),round(r[2],3)])
print(f"\n오염0% 점령 {rows[0][1]:.3f} -> 오염60% {rows[3][1]:.3f}: 허위보고가 코디네이터를 눈멀게 함")
print("Saved summary_consensus_attack.csv")
