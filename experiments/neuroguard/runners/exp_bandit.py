# -*- coding: utf-8 -*-
"""Learned router: contextual epsilon-greedy bandit over modes {retake, block, allow}, context =
(fast?, jam?), reward = episode 곱셈종합 (대회식). Train on random combo attacks, eval on suite."""
import sys, os, itertools
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg = yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml", encoding="utf-8"))
OUT = r"C:\workspace\DAH2026_exp"; COMP_F1 = 0.866; THRESH = 4
VEC_AIDS = {"W":[2,6],"J":[7,8],"B":[9,10],"K":[3,3],"F":[7,8]}; JAM_VECS={"J","F"}
MODES = ["retake","block","allow"]
def make_combo_red(vectors):
    class R(brains._Red):
        VECS=list(vectors)
        def get_action(self, obs, asp):
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5,obs)
            own=int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(VEC_AIDS[self.VECS[own%len(self.VECS)]])),obs)
    return R
ATTACKS=[("웜",["W"]),("ExploitKnown",["K"]),("재밍",["J"]),("FloodAll",["F"]),("차단",["B"]),("동시",["W","J","B"])]
def frontier(i,comp,pos,ml): return any(d!=i and np.linalg.norm(pos[i]-pos[d])<ml for d in comp)
def mode_aid(mode,i,comp,fr,jam):
    if i in comp: return 3
    if mode=="block": return 6 if fr else (4 if comp else 1)
    if mode=="retake": return 4 if comp else ("dejam" if jam else 1)
    return 4 if comp else ("dejam" if jam else 7)
def rollout(seed,vectors,Q,eps,rng_pol,learn):
    red=make_combo_red(vectors)
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,red); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    comp_frac,avail_frac,new_window=[],[],[]; prev=set(run.compromised_drones(cyborg,n)); k=len(vectors)
    used=set()
    for t in range(cfg["steps"]):
        comp=run.compromised_drones(cyborg,n); comp_frac.append(len(comp)/n)
        new=comp-prev; new_window.append(len(new))
        if len(new_window)>3: new_window.pop(0)
        fast=sum(new_window)>=THRESH; prev=set(comp)
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]
        red_jam=sum(1 for i in comp if vectors[i%k] in JAM_VECS); jam=red_jam>0
        cstate=(int(fast),int(jam))
        if rng_pol.random()<eps: mode=MODES[rng_pol.integers(len(MODES))]
        else: mode=MODES[int(np.argmax(Q[cstate]))]
        used.add((cstate,MODES.index(mode)))
        ctx={"compromised":comp,"ip_to_drone":ip2d,"n":n}; live=[a for a in env.active_agents if a in env.agent_actions]
        blocks=0;dejam=0;acts={}
        for a in live:
            i=int(a.split("_")[-1]); fr=frontier(i,comp,pos,ml); aid=mode_aid(mode,i,comp,fr,jam)
            if aid=="dejam": dejam+=1; acts[a]=actions.make_blue_index(1,env,a,ctx)
            else:
                if aid==6: blocks+=1
                acts[a]=actions.make_blue_index(aid,env,a,ctx)
        avail_frac.append(max(0.0,(n-len(comp)-blocks-red_jam+min(dejam,red_jam))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=comp_frac[-1]; auc=float(np.mean(comp_frac)); av=float(np.mean(avail_frac))
    reward=float(np.mean([1-final,1-auc,COMP_F1]))*av
    if learn:
        for (cs,mi) in used:
            Q[cs][mi]+= 0.1*(reward-Q[cs][mi])
    return final,av,reward
# train
Q={(f,j):np.zeros(len(MODES)) for f in (0,1) for j in (0,1)}
rng_pol=np.random.default_rng(7); rng_atk=np.random.default_rng(11)
fam=[list(c) for k in (1,2,3) for c in itertools.combinations("WJB",k)]
print("training bandit..."); 
for ep in range(240):
    vec=fam[int(rng_atk.integers(len(fam)))]; seed=3000+int(rng_atk.integers(50))
    rollout(seed,vec,Q,0.2,rng_pol,True)
print("learned policy (context -> best mode):")
for cs in Q: print(f"  fast={cs[0]} jam={cs[1]} -> {MODES[int(np.argmax(Q[cs]))]}  Q={np.round(Q[cs],3)}")
# eval
EVAL=[3000,3001,3002,3003,3004]; fc_g={};m_g={}
print("\n=== bandit eval on suite ===")
for an,vec in ATTACKS:
    rs=[rollout(s,vec,Q,0.0,rng_pol,False) for s in EVAL]
    fc=float(np.mean([r[0] for r in rs])); m=float(np.mean([r[2] for r in rs])); fc_g[an]=fc;m_g[an]=m
    print(f"  {an:12} 점령 {fc:.3f} | 곱셈 {m:.3f}")
print(f"\nbandit: 평균점령 {np.mean(list(fc_g.values())):.3f} | worst {max(fc_g.values()):.3f} | 곱셈종합 {np.mean(list(m_g.values())):.3f}")
with open(os.path.join(OUT,"summary_bandit.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["attack","점령","곱셈"]); 
    for an in fc_g: wr.writerow([an,round(fc_g[an],3),round(m_g[an],3)])
print("Saved summary_bandit.csv")
