# -*- coding: utf-8 -*-
"""A1. Role-allocation COORDINATOR (CTDE): assign each compromised node to ONE nearest clean drone
to retake (no redundant collisions), rest do availability. vs uncoordinated (all retake first node)."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT=r"C:\workspace\DAH2026_exp"; EVAL=[3000,3001,3002,3003,3004]; COMP_F1=0.866
VEC_AIDS={"W":[2,6],"J":[7,8],"B":[9,10],"K":[3,3],"F":[7,8]}; JAM_VECS={"J","F"}
def make_combo_red(v):
    class R(brains._Red):
        VECS=list(v)
        def get_action(self,obs,asp):
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5,obs)
            o=int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[o%len(self.VECS)]])),obs)
    return R
ATTACKS=[("웜",make_combo_red(["W"]),["W"]),("ExploitKnown",make_combo_red(["K"]),["K"]),
         ("재밍",make_combo_red(["J"]),["J"]),("동시",make_combo_red(["W","J","B"]),["W","J","B"]),("rule웜",brains.RuleRed,None)]
def retake_target(env,a,node,ip2d,sleep):
    idx=actions.action_index_map(env,a)
    for i,ip in idx.get("RetakeControl",[]):
        if ip2d.get(ip)==node: return i
    c=idx.get("RetakeControl",[]); return c[0][0] if c else sleep
def rollout(seed,red,coord,vectors):
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,red); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    cf,af=[],[]; k=len(vectors) if vectors else 1
    for t in range(cfg["steps"]):
        comp=run.compromised_drones(cyborg,n); cf.append(len(comp)/n)
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        red_jam=sum(1 for i in comp if vectors and vectors[i%k] in JAM_VECS)
        ctx={"compromised":comp,"ip_to_drone":ip2d,"n":n}
        live=[a for a in env.active_agents if a in env.agent_actions]
        sleep=actions.action_index_map(env,live[0]).get("Sleep",[(0,None)])[0][0] if live else 0
        acts={}; dejam=0
        if coord:
            clean=[int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp]
            used=set(); assign={}
            for c in comp:
                cand=sorted([d for d in clean if d not in used],key=lambda d:np.linalg.norm(pos[d]-pos[c]))
                if cand: assign[cand[0]]=c; used.add(cand[0])
            for a in live:
                i=int(a.split("_")[-1])
                if i in comp: acts[a]=actions.make_blue_index(3,env,a,ctx)
                elif i in assign: acts[a]=retake_target(env,a,assign[i],ip2d,sleep)
                else:
                    if red_jam>0: dejam+=1; acts[a]=actions.make_blue_index(1,env,a,ctx)
                    else: acts[a]=actions.make_blue_index(1,env,a,ctx)
        else:
            for a in live:
                i=int(a.split("_")[-1])
                if i in comp: acts[a]=actions.make_blue_index(3,env,a,ctx)
                elif comp:
                    if red_jam>0: dejam+=1; acts[a]=actions.make_blue_index(1,env,a,ctx)
                    else: acts[a]=actions.make_blue_index(4,env,a,ctx)   # all retake first compromised (collision)
                else: acts[a]=actions.make_blue_index(1,env,a,ctx)
        af.append(max(0.0,(n-len(comp)-red_jam+min(dejam,red_jam))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=cf[-1]; auc=float(np.mean(cf)); av=float(np.mean(af))
    return final, av, float(np.mean([1-final,1-auc,COMP_F1]))*av
def ev(red,coord,v):
    rs=[rollout(s,red,coord,v) for s in EVAL]; return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))
print("=== A1 coordinator vs uncoordinated (점령 | 가용성 | 곱셈) ===")
g={}
for an,red,v in ATTACKS:
    u=ev(red,False,v); c=ev(red,True,v); g[an]=(u,c)
    print(f"[{an:12}] uncoord {u[0]:.3f}/{u[2]:.3f}  | coord {c[0]:.3f}/{c[2]:.3f}")
names=list(g)
uc=(np.mean([g[a][0][0] for a in names]),max(g[a][0][0] for a in names),np.mean([g[a][0][2] for a in names]))
co=(np.mean([g[a][1][0] for a in names]),max(g[a][1][0] for a in names),np.mean([g[a][1][2] for a in names]))
print(f"\nuncoord: 평균점령 {uc[0]:.3f} worst {uc[1]:.3f} 곱셈 {uc[2]:.3f}")
print(f"coord  : 평균점령 {co[0]:.3f} worst {co[1]:.3f} 곱셈 {co[2]:.3f}")
with open(os.path.join(OUT,"summary_coordinator.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["mode","평균점령","worst","곱셈"]); wr.writerow(["uncoord",*[round(x,3) for x in uc]]); wr.writerow(["coord",*[round(x,3) for x in co]])
print("Saved summary_coordinator.csv")
