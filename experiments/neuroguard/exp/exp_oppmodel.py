# -*- coding: utf-8 -*-
"""C7. Opponent-aware MPC: the self-simulation model ESTIMATES the attacker (spread rate beta from
observed spread, jam load) instead of a fixed generic worm model, then best-responds. vs fixed MPC."""
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
def frontier(i,comp,pos,ml): return any(d!=i and np.linalg.norm(pos[i]-pos[d])<ml for d in comp)
def adjacency(pos,ml):
    d=np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=-1); return (d<ml)&(d>0)
MODE_PARAM={"retake":(1.0,0.5,0.05),"block":(0.4,0.2,0.40),"allow":(1.0,0.0,0.0)}
def mpc_mode(comp,A,n,beta,jam_frac,opp,k=3):
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
        comp_pred=float(p.mean())
        # opponent-aware: jamming hurts availability regardless of mode; block can't stop jam
        jam_pen=jam_frac if opp else 0.0
        avail_pred=max(0.0,1-comp_pred-bc-jam_pen)
        val=(1-comp_pred)*avail_pred
        if val>bv: bv,best=val,mode
    return best
def mode_aid(mode,i,comp,fr,jam):
    if i in comp: return 3
    if mode=="block": return 6 if fr else (4 if comp else 1)
    if mode=="retake": return 4 if comp else ("dejam" if jam else 1)
    return 4 if comp else ("dejam" if jam else 7)
def rollout(seed,red,opp,vectors):
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,red); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    cf,af,nw=[],[],[]; prev=set(run.compromised_drones(cyborg,n)); k=len(vectors) if vectors else 1
    for t in range(cfg["steps"]):
        comp=run.compromised_drones(cyborg,n); cf.append(len(comp)/n)
        new=comp-prev; nw.append(len(new))
        if len(nw)>3: nw.pop(0)
        prev=set(comp); pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        red_jam=sum(1 for i in comp if vectors and vectors[i%k] in JAM_VECS); jam=red_jam>0
        if opp:  # estimate beta from observed spread
            denom=max(1,len(comp)); beta=float(np.clip(sum(nw)/(3*denom),0.05,0.6))
        else:
            beta=0.35
        mode=mpc_mode(comp,adjacency(pos,ml),n,beta,red_jam/n,opp) if comp else "retake"
        ctx={"compromised":comp,"ip_to_drone":ip2d,"n":n}; live=[a for a in env.active_agents if a in env.agent_actions]
        blocks=0;dejam=0;acts={}
        for a in live:
            i=int(a.split("_")[-1]); fr=frontier(i,comp,pos,ml); aid=mode_aid(mode,i,comp,fr,jam)
            if aid=="dejam": dejam+=1; acts[a]=actions.make_blue_index(1,env,a,ctx)
            else:
                if aid==6: blocks+=1
                acts[a]=actions.make_blue_index(aid,env,a,ctx)
        af.append(max(0.0,(n-len(comp)-blocks-red_jam+min(dejam,red_jam))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=cf[-1]; auc=float(np.mean(cf)); av=float(np.mean(af))
    return final,av,float(np.mean([1-final,1-auc,COMP_F1]))*av
def ev(red,opp,v): rs=[rollout(s,red,opp,v) for s in EVAL]; return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))
print("=== C7 opponent-aware MPC vs fixed MPC (점령 | 곱셈) ===")
g={}
for an,red,v in ATTACKS:
    fx=ev(red,False,v); op=ev(red,True,v); g[an]=(fx,op)
    print(f"[{an:12}] fixed {fx[0]:.3f}/{fx[2]:.3f} | opp-aware {op[0]:.3f}/{op[2]:.3f}")
names=list(g)
fx=(np.mean([g[a][0][0] for a in names]),max(g[a][0][0] for a in names),np.mean([g[a][0][2] for a in names]))
op=(np.mean([g[a][1][0] for a in names]),max(g[a][1][0] for a in names),np.mean([g[a][1][2] for a in names]))
print(f"\nfixed MPC   : 평균점령 {fx[0]:.3f} worst {fx[1]:.3f} 곱셈 {fx[2]:.3f}")
print(f"opp-aware MPC: 평균점령 {op[0]:.3f} worst {op[1]:.3f} 곱셈 {op[2]:.3f}")
with open(os.path.join(OUT,"summary_oppmodel.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["mode","평균점령","worst","곱셈"]); wr.writerow(["fixed_mpc",*[round(x,3) for x in fx]]); wr.writerow(["opp_mpc",*[round(x,3) for x in op]])
print("Saved summary_oppmodel.csv")
