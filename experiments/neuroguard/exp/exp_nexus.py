# -*- coding: utf-8 -*-
"""FINAL unified defense NEXUS = role-allocation COORDINATOR + MPC self-sim posture + availability-
center (de-jam, minimal block). Also sweep satellite BLACKOUT p to see coordination degrade.
Team-aligned metric (comp_F1=0.866)."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg=yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml",encoding="utf-8"))
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
         ("재밍",make_combo_red(["J"]),["J"]),("FloodAll",make_combo_red(["F"]),["F"]),
         ("차단",make_combo_red(["B"]),["B"]),("동시",make_combo_red(["W","J","B"]),["W","J","B"]),("rule웜",brains.RuleRed,None)]
def adjacency(pos,ml):
    d=np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=-1); return (d<ml)&(d>0)
MODE_PARAM={"retake":(1.0,0.5,0.05),"block":(0.4,0.2,0.40),"allow":(1.0,0.0,0.0)}
def mpc_mode(comp,A,n,beta=0.35,k=3):
    best,bv="retake",-1
    for mode,(ss,rt,bc) in MODE_PARAM.items():
        p=np.zeros(n)
        for i in comp: p[i]=1.0
        be=beta*ss
        for _ in range(k):
            np2=p.copy()
            for j in range(n):
                if p[j]<1:
                    pr=float(np.prod(1-be*p[A[j]])) if A[j].any() else 1.0
                    np2[j]=p[j]+(1-p[j])*(1-pr)
            p=np2*(1-rt)
        cp=float(p.mean()); av=max(0.0,1-cp-bc); val=(1-cp)*av
        if val>bv: bv,best=val,mode
    return best
def target_index(env,a,node,ip2d,cls,sleep):
    idx=actions.action_index_map(env,a)
    for i,ip in idx.get(cls,[]):
        if ip2d.get(ip)==node: return i
    c=idx.get(cls,[]); return c[0][0] if c else sleep
def rollout(seed,red,defense,vectors,p_black=0.0):
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,red); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    rng=np.random.default_rng(seed+321); kb=int(round(p_black*n))
    black=set(int(x) for x in rng.choice(n,size=kb,replace=False)) if kb else set()
    cf,af=[],[]; k=len(vectors) if vectors else 1
    for t in range(cfg["steps"]):
        comp_true=run.compromised_drones(cyborg,n); cf.append(len(comp_true)/n)
        comp=comp_true-black                          # defender sees only connected
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        red_jam=sum(1 for i in comp_true if vectors and vectors[i%k] in JAM_VECS)
        jam=red_jam>0
        ctx={"compromised":comp,"ip_to_drone":ip2d,"n":n}
        live=[a for a in env.active_agents if a in env.agent_actions]
        sleep=actions.action_index_map(env,live[0]).get("Sleep",[(0,None)])[0][0] if live else 0
        mode = mpc_mode(comp,adjacency(pos,ml),n) if (defense=="nexus" and comp) else "retake"
        clean=[int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp_true and int(a.split("_")[-1]) not in black]
        assign={}; used=set()
        if defense in ("coord","nexus"):
            for c in comp:
                cand=sorted([d for d in clean if d not in used],key=lambda d:np.linalg.norm(pos[d]-pos[c]))
                if cand: assign[cand[0]]=c; used.add(cand[0])
        blocks=0; dejam=0; acts={}
        for a in live:
            i=int(a.split("_")[-1])
            if i in black: acts[a]=actions.make_blue_index(0,env,a,ctx); continue
            if i in comp_true: acts[a]=actions.make_blue_index(3,env,a,ctx); continue
            if i in assign:
                if defense=="nexus" and mode=="block":
                    blocks+=1; acts[a]=target_index(env,a,assign[i],ip2d,"BlockTraffic",sleep)
                else:
                    acts[a]=target_index(env,a,assign[i],ip2d,"RetakeControl",sleep)
            else:
                if jam: dejam+=1; acts[a]=actions.make_blue_index(1,env,a,ctx)
                elif defense=="nexus" and mode=="allow": acts[a]=actions.make_blue_index(7,env,a,ctx)
                else: acts[a]=actions.make_blue_index(1,env,a,ctx)
        af.append(max(0.0,(n-len(comp_true)-blocks-red_jam+min(dejam,red_jam))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=cf[-1]; auc=float(np.mean(cf)); av=float(np.mean(af))
    return final,av,float(np.mean([1-final,1-auc,COMP_F1]))*av
def ev(red,defense,v,p=0.0):
    rs=[rollout(s,red,defense,v,p) for s in EVAL]; return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))

print("=== NEXUS (coord+MPC+가용성) vs coord vs mpc-only (점령 | 가용성 | 곱셈) ===")
g={}
for an,red,v in ATTACKS:
    co=ev(red,"coord",v); nx=ev(red,"nexus",v); g[an]=(co,nx)
    print(f"[{an:12}] coord {co[0]:.3f}/{co[2]:.3f} | NEXUS {nx[0]:.3f}/{nx[2]:.3f}")
names=list(g)
for label,idx in [("coord",0),("NEXUS",1)]:
    avgfc=np.mean([g[a][idx][0] for a in names]); worst=max(g[a][idx][0] for a in names); avgm=np.mean([g[a][idx][2] for a in names])
    print(f"{label:7}: 평균점령 {avgfc:.3f} worst {worst:.3f} 곱셈 {avgm:.3f}")

print("\n=== NEXUS x 위성 단절 p (동시공격, 점령) — 협조 저하 ===")
PS=[0.0,0.2,0.4,0.6]; red_s=make_combo_red(["W","J","B"]); v_s=["W","J","B"]
print("p".ljust(6)+"coord    NEXUS")
sweep=[]
for p in PS:
    c=ev(red_s,"coord",v_s,p)[0]; x=ev(red_s,"nexus",v_s,p)[0]; sweep.append((p,c,x))
    print(f"{p:<6.1f}{c:.3f}   {x:.3f}")
with open(os.path.join(OUT,"summary_nexus.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["attack","coord_점령","coord_곱셈","nexus_점령","nexus_곱셈"])
    for a in names: wr.writerow([a,round(g[a][0][0],3),round(g[a][0][2],3),round(g[a][1][0],3),round(g[a][1][2],3)])
    wr.writerow([]); wr.writerow(["blackout_p","coord_동시","nexus_동시"])
    for r in sweep: wr.writerow([r[0],round(r[1],3),round(r[2],3)])
print("Saved summary_nexus.csv")
