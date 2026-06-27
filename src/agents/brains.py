"""The three agent types for both red and blue, over one shared action catalog.

  rule : hand-written heuristic / FSM  (the report's strong baseline)
  llm  : LLM picks from a text menu     (offline deterministic stub by default,
                                         Claude when ANTHROPIC_API_KEY is set)
  rl   : frozen tabular Q-policy        (trained & cached by rl.py)

Red agents are BaseAgent subclasses passed to the sim as default_red_agent.
Blue is a per-step selector returning a catalog decision id (0-7), mapped to a
wrapper action index by actions.make_blue_index().
"""
from CybORG.Agents import BaseAgent

from agents import actions, llm

# ----- shared state for the rl agents (set by rl.py / sweep.py) -----
_BLUE_Q = None
_RED = {"q": None, "eps": 0.0, "rng": None, "train": False}
RED_STASH = []                       # (name, state, aid) collected during rl-red training


def use_rl(q_blue, q_red):
    """Install frozen Q-tables for greedy evaluation."""
    global _BLUE_Q
    _BLUE_Q = q_blue
    set_red_q(q_red, 0.0, None, train=False)


def set_red_q(q, eps, rng, train):
    _RED.update(q=q, eps=eps, rng=rng, train=train)


def pop_red_stash():
    global RED_STASH
    s, RED_STASH = RED_STASH, []
    return s


# ===================================================================== RED ===
def red_state(obs, mem, t=0):
    phase = 0 if t < 8 else (1 if t < 20 else 2)          # early / mid / late episode
    return (1 if mem.get("target") is not None else 0,
            1 if obs.get("success") is True else 0, phase)


class _Red(BaseAgent):
    def __init__(self, name, np_random=None):
        super().__init__(name, np_random)
        self.mem, self.t = {}, 0

    def train(self, results):
        pass

    def end_episode(self):
        self.mem, self.t = {}, 0

    def set_initial_values(self, action_space, observation):
        pass

    def _emit(self, aid, obs):
        action, _ = actions.make_red_action(aid, obs, self.name, self.mem, self.np_random)
        self.t += 1
        return action


class RuleRed(_Red):
    """Exploit-then-seize spreader with occasional jamming (RedDroneWorm-style)."""
    def get_action(self, obs, action_space):
        if self.mem.get("target") is not None and obs.get("success") is True:
            aid = 5                                       # SeizeControl
        else:
            aid = int(self.np_random.choice([2, 2, 3, 7, 10]))  # exploit / jam / persist
        return self._emit(aid, obs)


class LLMRed(_Red):
    """LLM chooses an attack from the menu; offline stub spreads wide + jams."""
    def get_action(self, obs, action_space):
        nvis = len(actions.ip_list(obs))

        def stub():
            if self.mem.get("target") is not None and obs.get("success") is True:
                return 5                                  # finish the takeover
            if nvis == 0:
                return 1                                  # discover
            return 4 if self.t % 3 else 7                 # farthest-exploit / jam

        prompt = (f"You are a red (attacker) agent on a drone swarm network.\n"
                  f"Visible drones: {nvis}. Last action success: {obs.get('success')}. "
                  f"Have an exploited target ready to seize: "
                  f"{self.mem.get('target') is not None}.\n"
                  f"Pick the best NEXT attack action by id:\n{actions.red_menu_text()}")
        aid = llm.choose(prompt, range(actions.RED_N), stub)
        return self._emit(aid, obs)


class RLRed(_Red):
    """Frozen tabular Q-policy (greedy) over (have_target, last_success)."""
    def get_action(self, obs, action_space):
        s = red_state(obs, self.mem, self.t)
        q = _RED["q"]
        aid = (int(self.np_random.randint(0, actions.RED_N)) if q is None
               else q.act(s, _RED["eps"], _RED["rng"]))
        if _RED["train"]:
            RED_STASH.append((self.name, s, aid))
        return self._emit(aid, obs)


RED_BRAINS = {"rule": RuleRed, "llm": LLMRed, "rl": RLRed}


# ==================================================================== BLUE ===
def blue_state(agent, ctx):
    own = int(agent.split("_")[-1])
    frac = len(ctx["compromised"]) / max(1, ctx["n"])
    return (1 if own in ctx["compromised"] else 0, min(3, int(frac * 4)))


def blue_decide(btype, env, agent, ctx):
    """Return a blue catalog decision id (0-7) for one agent this step."""
    own = int(agent.split("_")[-1])
    comp = ctx["compromised"]

    if btype == "rl":
        if _BLUE_Q is None:
            return 1
        return _BLUE_Q.act(blue_state(agent, ctx), 0.0, None)

    if btype == "llm":
        def stub():
            if own in comp:
                return 3                                  # remove sessions on own
            if comp:
                return 6                                  # block a compromised drone
            return 2                                      # analyse proactively

        prompt = (f"You are a blue (defender) agent for drone {own} on a swarm "
                  f"network.\nOwn drone compromised: {own in comp}. "
                  f"Total compromised drones: {len(comp)}/{ctx['n']}.\n"
                  f"Pick the best NEXT defensive action by id:\n{actions.blue_menu_text()}")
        return llm.choose(prompt, range(actions.BLUE_DECISION_N), stub)

    # rule
    if own in comp:
        return 3                                          # remove sessions on own
    if comp:
        return 4                                          # retake a compromised drone
    return 1                                              # monitor
