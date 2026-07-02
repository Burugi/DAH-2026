# -*- coding: utf-8 -*-
"""Central pair closing the 'connectivity=ceiling' thesis:
  ATTACK frag : jam the top-K hub/bridge nodes -> graph partitions -> isolated drones become
                unreachable reservoirs (dynamic, attack-driven blackout).
  DEFENSE relay: spare clean drones reposition to re-bridge -> shrink the isolated set (raise ceiling).
  + predictive coordinator: assign defenders to PREDICTED frontier (worm's next hop), not just current.
Team-aligned metric (comp_F1=0.866)."""
import sys, os
sys.path.insert(0, r"C:\workspace\DAH-2026\src"); os.chdir(r"C:\workspace\DAH-2026")
import numpy as np, yaml, csv
import run
from agents import brains, actions
cfg=yaml.safe_load(open(r"C:\workspace\DAH-2026\src\configs\sweep.yaml",encoding="utf-8"))
OUT=r"C:\workspace\DAH2026_exp"; EVAL=[3000,3001,3002,3003,3004]; COMP_F1=0.866
def make_combo_red(v):
    A={"W":[2,6],"J":[7,8],"B":[9,10]}
    class R(brains._Red):
        VECS=list(v)
        def get_action(self,obs,asp):
            if self.mem.get("target") is not None and obs.get("success") is True: return self._emit(5,obs)
            o=int(self.name.split("_")[-1]); return self._emit(int(self.np_random.choice(A[self.VECS[o%len(self.VECS)]])),obs)
    return R
def adjacency(pos,ml):
    d=np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=-1); return (d<ml)&(d>0)
def largest_comp(present,A,n):
    seen=set(); best=set()
    for s in present:
        if s in seen: continue
        comp=set(); st=[s]
        while st:
            u=st.pop()
            if u in seen: continue
            seen.add(u); comp.add(u)
            for v in range(n):
                if A[u,v] and v in present and v not in seen: st.append(v)
        if len(comp)>len(best): best=comp
    return best
def retake_target(env,a,node,ip2d,sleep):
    idx=actions.action_index_map(env,a)
    for i,ip in idx.get("RetakeControl",[]):
        if ip2d.get(ip)==node: return i
    c=idx.get("RetakeControl",[]); return c[0][0] if c else sleep
def frontier_nodes(comp,A,n,present):
    fr=set()
    for c in comp:
        for j in range(n):
            if A[c,j] and j not in comp and j in present: fr.add(j)
    return fr
def rollout(seed,defense,frag_K,relay_R):
    fleet,cyborg,env,ip2d=run.build_env(cfg,seed,make_combo_red(["W","J","B"])); n=fleet["n"]; ml=cfg["fleet"].get("max_link",40)
    pos0=fleet["pos_true"][0]; deg=adjacency(pos0,ml).sum(1)
    hubs=set(int(x) for x in np.argsort(-deg)[:frag_K]) if frag_K else set()   # attacker jams these bridges
    cf,af=[],[]
    for t in range(cfg["steps"]):
        comp=run.compromised_drones(cyborg,n); cf.append(len(comp)/n)
        pos=fleet["pos_true"][min(t,fleet["steps"]-1)]; A=adjacency(pos,ml)
        present=set(range(n))-hubs                         # jammed hubs can't relay
        big=largest_comp(present,A,n) if present else set()
        isolated=(present-big)                             # partitioned off the giant component
        if relay_R and isolated:                           # relay drones re-bridge some isolated
            for x in list(isolated)[:relay_R]: isolated.discard(x)
        unreachable=hubs|isolated                          # defender can't coordinate these
        reachable_comp=comp-unreachable
        ctx={"compromised":comp,"ip_to_drone":ip2d,"n":n}
        live=[a for a in env.active_agents if a in env.agent_actions]
        sleep=actions.action_index_map(env,live[0]).get("Sleep",[(0,None)])[0][0] if live else 0
        clean=[int(a.split("_")[-1]) for a in live if int(a.split("_")[-1]) not in comp and int(a.split("_")[-1]) not in unreachable]
        assign={}; used=set()
        targets=list(reachable_comp)
        if defense=="coord_pred":                          # also pre-guard frontier (predicted next hop)
            targets = targets + list(frontier_nodes(reachable_comp,A,n,present-set(comp)))
        for c in targets:
            cand=sorted([d for d in clean if d not in used],key=lambda d:np.linalg.norm(pos[d]-pos[c]))
            if cand: assign[cand[0]]=c; used.add(cand[0])
        acts={}; dejam=0
        for a in live:
            i=int(a.split("_")[-1])
            if i in unreachable: acts[a]=actions.make_blue_index(0,env,a,ctx); continue
            if i in comp: acts[a]=actions.make_blue_index(3,env,a,ctx); continue
            if i in assign and assign[i] in comp: acts[a]=retake_target(env,a,assign[i],ip2d,sleep)
            elif i in assign: acts[a]=actions.make_blue_index(6,env,a,ctx)   # frontier guard -> block worm hop
            else: acts[a]=actions.make_blue_index(1,env,a,ctx)
        af.append(max(0.0,(n-len(comp)-len(unreachable))/n))
        _,rew,done,_=env.step(acts)
        if all(done.values()): break
    final=cf[-1]; auc=float(np.mean(cf)); av=float(np.mean(af))
    return final,av,float(np.mean([1-final,1-auc,COMP_F1]))*av
def ev(defense,K,R):
    rs=[rollout(s,defense,K,R) for s in EVAL]; return tuple(float(np.mean([r[j] for r in rs])) for j in range(3))
print("=== predictive coordinator (정상 동시공격) ===")
for d in ["coord","coord_pred"]:
    fc,av,m=ev(d,0,0); print(f"  {d:11} 점령 {fc:.3f} | 가용성 {av:.3f} | 곱셈 {m:.3f}")
print("\n=== 연결성 단절 공격(K 허브 재밍) x 복원 방어(relay) — 동시공격, 점령|곱셈 ===")
print("K".ljust(4)+"  coord(복원X)        coord+relay(복원O)")
rows=[]
for K in [0,2,4,6]:
    a=ev("coord",K,0); b=ev("coord",K,max(2,K))
    rows.append((K,a,b)); print(f"{K:<4}  {a[0]:.3f}/{a[2]:.3f}          {b[0]:.3f}/{b[2]:.3f}")
with open(os.path.join(OUT,"summary_connectivity.csv"),"w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["frag_K","coord_점령","coord_곱셈","relay_점령","relay_곱셈"])
    for K,a,b in rows: wr.writerow([K,round(a[0],3),round(a[2],3),round(b[0],3),round(b[2],3)])
print("Saved summary_connectivity.csv")
