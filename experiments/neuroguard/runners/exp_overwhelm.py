# -*- coding: utf-8 -*-
"""(a) Anti-coordination OVERWHELM: synchronized burst (S initial red seeds) > defender capacity.
   (b) Low-and-slow EVASION: attacker tempo under detection threshold vs coordinator.
Coordinator defense. Team metric."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg=yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml",encoding="utf-8"))
OUT=r"C:\workspace\DAH2026_exp"; EVAL=[3000,3001,3002,3003,3004]; COMP_F1=0.866
def make_tempo_red(tempo):
    class R(brains._Red):
        def get_action(self,obs,asp):
            if self.np_random.uniform()>tempo: return self._emit(0,obs)
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5,obs)
            return self._emit(int(self.np_random.choice([2,6])),obs)
    return R
def retake_target(env,a,node,ip2d,sleep):
    idx=actions.action_index_map(env,a)
    for i,ip in idx.get("RetakeControl",[]):
        if ip2d.get(ip)==node: return i
    c=idx.get("RetakeControl",[]); return c[0][0] if c else sleep
def rollout(seed,red,start_red):
    cfg2=dict(cfg); cfg2["sim"]=dict(cfg["sim"]); cfg2["sim"]["starting_num_red"]=start_red
    fleet,cyborg,env,ip2d=run.build_env(cfg2,seed,red); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    cf,af=[],[]
    for t in range(cfg["steps"]):
        comp=run.compromised_drones(cyborg,n); cf.append(len(comp)/n)
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        ctx={"compromised":comp,"ip_to_drone":ip2d,"n":n}
        live=[a for a in env.active_agents if a in env.agent_actions]
        sleep=actions.action_index_map(env,live[0]).get("Sleep",[(0,None)])[0][0] if live else 0
        clean=[int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
        assign={}; used=set()
        for c in comp:
            cand=sorted([d for d in clean if d not in used],key=lambda d:np.linalg.norm(pos[d]-pos[c]))
            if cand: assign[cand[0]]=c; used.add(cand[0])
        acts={}
        for a in live:
            i=int(a.split("_")[-1])
            if i in comp: acts[a]=actions.make_blue_index(3,env,a,ctx)
            elif i in assign: acts[a]=retake_target(env,a,assign[i],ip2d,sleep)
            else: acts[a]=actions.make_blue_index(1,env,a,ctx)
        af.append(max(0.0,(n-len(comp))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=cf[-1]; auc=float(np.mean(cf)); av=float(np.mean(af))
    return final,float(np.mean([1-final,1-auc,COMP_F1]))*av
def ev(red,S): rs=[rollout(s,red,S) for s in EVAL]; return tuple(float(np.mean([r[j] for r in rs])) for j in range(2))
print("=== (a) 반협조 폭주: 동시 시드 S개 (코디네이터, 점령|곱셈) ===")
print("S".ljust(4)+"점령     곱셈")
rb=[]
for S in [1,2,4,6]:
    fc,m=ev(make_tempo_red(1.0),S); rb.append((S,fc,m)); print(f"{S:<4}{fc:.3f}   {m:.3f}")
print("\n=== (b) 저강도 잠행: tempo 낮춰 회피 (코디네이터, 점령) ===")
print("tempo".ljust(7)+"점령")
rt=[]
for tp in [0.2,0.4,0.6,0.8,1.0]:
    fc,m=ev(make_tempo_red(tp),1); rt.append((tp,fc)); print(f"{tp:<7.1f}{fc:.3f}")
with open(os.path.join(OUT,"summary_overwhelm.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["start_red_S","점령","곱셈"]);
    for r in rb: wr.writerow([r[0],round(r[1],3),round(r[2],3)])
    wr.writerow([]); wr.writerow(["tempo","점령"])
    for r in rt: wr.writerow([r[0],round(r[1],3)])
print("Saved summary_overwhelm.csv")
