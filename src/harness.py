"""Defense-policy harness for the HVT experiment.

Provides the graph/utility primitives and the DefensePolicy base class that
src/agents/hvt.py builds on, plus a vector-driven red agent factory. run_hvt.py
drives a DefensePolicy over the DroneSwarm sim using these.

  vectors : attack lanes per scenario — W (worm exploit+spread) / J (jam) / B (block)
  A DefensePolicy sees only detected+adjacency, never the true compromise set.
"""
import numpy as np

from agents import actions, brains

# attack-lane -> red catalog action ids (worm / jam / block)
VEC_AIDS = {"W": [2, 6], "J": [7, 8], "B": [9, 10]}
JAM_VECS = {"J", "B"}


def adjacency(pos, ml):
    """Boolean proximity graph: drones within max-link distance `ml` (no self-edge)."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    return (d < ml) & (d > 0)


def components(present, A, n):
    """Connected components (as sets of node ids) among `present` nodes over graph A."""
    seen, out = set(), []
    for s in present:
        if s in seen:
            continue
        cc, st = set(), [s]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u)
            cc.add(u)
            for v in range(n):
                if A[u, v] and v in present and v not in seen:
                    st.append(v)
        out.append(cc)
    return out


def retake_target(env, a, node, ip2d, sleep):
    """Wrapper action index that retakes drone `node`, or the first RetakeControl / Sleep."""
    idx = actions.action_index_map(env, a)
    for i, ip in idx.get("RetakeControl", []):
        if ip2d.get(ip) == node:
            return i
    c = idx.get("RetakeControl", [])
    return c[0][0] if c else sleep


def make_red(vectors, tempo):
    """Red agent class emitting the scenario's attack lanes at the given tempo."""
    class R(brains._Red):
        VECS = list(vectors)

        def get_action(self, obs, action_space):
            if self.np_random.uniform() > tempo:
                return self._emit(0, obs)
            if self.mem.get("target") is not None and obs.get("success") is True:
                return self._emit(5, obs)
            o = int(self.name.split("_")[-1])
            lane = self.VECS[o % len(self.VECS)]
            return self._emit(int(self.np_random.choice(VEC_AIDS[lane])), obs)
    return R


class DefensePolicy:
    """Base class for the swarm defense policies. Subclasses implement reset + step.

    step(comp, pos, env, live, ip2d, rng) -> (acts, availability)
      comp  : true compromised drone ids (for detector sampling / accounting only)
      pos   : drone positions this step        live : active blue agent names
      ip2d  : ip -> drone id map               returns per-agent action indices + avail
    """
    name = "base"

    def reset(self, cfg, fleet, spec, hubs, black, ml, recall, fp):
        raise NotImplementedError

    def step(self, comp, pos, env, live, ip2d, rng):
        raise NotImplementedError


def scenario_hubs_black(fleet, spec, ml, rng):
    """Derive fragmentation hubs (top-degree) and blackout nodes from a scenario spec."""
    n = fleet["n"]
    pos0 = fleet["pos_true"][0]
    hubs = set()
    if spec.get("frag_K"):
        deg = adjacency(pos0, ml).sum(1)
        hubs = {int(x) for x in np.argsort(-deg)[:spec["frag_K"]]}
    black = set()
    kbl = int(round(spec.get("blackout_p", 0.0) * n))
    if kbl:
        black = {int(x) for x in rng.choice(n, size=kbl, replace=False)}
    return hubs, black


_NAME2BLUE = {"RemoveOtherSessions": 3, "RetakeControl": 4,
              "BlockTraffic": 6, "AllowTraffic": 7}


def blue_action_tape(env, acts):
    """(rep_id, count_vector) over the blue catalog for the actions taken this step.

    Sleep-index actions (monitor / de-jam) count as Monitor. Used for the dashboard
    tactic log, mirroring run.py's per-step blue summary.
    """
    row = np.zeros(actions.BLUE_DECISION_N, np.int16)
    for a, widx in acts.items():
        action = env.agent_actions.get(a, {}).get(widx)
        name = type(action).__name__ if action is not None else "Sleep"
        row[_NAME2BLUE.get(name, 1)] += 1
    rep = int(row.argmax()) if row.sum() else 1
    return rep, row
