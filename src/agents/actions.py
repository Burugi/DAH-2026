"""Single source of truth for the red/blue action catalogs.

Every action is grounded in a real CybORG DroneSwarm primitive (or, for the
passive blue defences, in the synthetic `defense.py` layer) and tagged with a
MITRE ATT&CK / D3FEND id so the experiment is traceable to https://attack.mitre.org.

The three agent types (rule / llm / rl, see brains.py) all choose from these
same catalogs; only *how* they choose differs.

  - Red decisions  -> make_red_action()  builds a sim Action object.
  - Blue decisions -> make_blue_index()  picks an index in env.agent_actions[agent].

Catalog entries that don't change sim state directly (Discover / Monitor /
Analyse, and the passive blue defences 8-11) are still real *decisions* an agent
reasons about; they are logged so behavioural differences are visible.
"""
from math import sqrt

from CybORG.Simulator.Actions import (ExploitDroneVulnerability, FloodBandwidth,
                                       SeizeControl, Sleep)
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTraffic


# ---------------------------------------------------------------- catalogs ---
# (name, mitre tag, description). Index == action id used everywhere.
RED_CATALOG = [
    ("Sleep",          "TA0005 Defense Evasion",        "do nothing this step"),
    ("DiscoverDrones", "T1018 Remote System Discovery", "enumerate visible drones"),
    ("ExploitNearest", "T1210 Exploit Remote Services", "exploit the closest drone"),
    ("ExploitRandom",  "T1210 Exploit Remote Services", "exploit a random drone"),
    ("ExploitFarthest","T1210 Exploit Remote Services", "exploit the farthest drone"),
    ("SeizeControl",   "T1078 Valid Accounts + T1542 Persist","seize/hold a freshly exploited drone"),
    ("SpreadWorm",     "T1021 Lateral Movement",        "exploit a neighbour to spread"),
    ("JamNearest",     "T1498 Network DoS",             "flood/jam the closest drone"),
    ("JamFarthest",    "T1499 Endpoint DoS",            "flood/jam the farthest drone"),
    ("BlockComms",     "T1565 Service Stop",            "block a drone's traffic"),
    ("Persist",        "T1542 Firmware/Boot Persistence","re-assert control over an owned drone"),
]

# 0-8 are blue decisions an agent picks each step.
# 9-12 are passive telemetry defences enabled via the scenario `defense` block
# and scored by defense.py (kept here for a complete, MITRE-mapped catalog).
BLUE_CATALOG = [
    ("Sleep",            "—",                                "do nothing this step"),
    ("Monitor",          "D3FEND Network Traffic Analysis",  "observe events (no state change)"),
    ("Analyse",          "D3FEND Process/File Analysis",     "inspect sessions on own drone"),
    ("RemoveSessions",   "D3FEND Process Termination/M1018", "kill red sessions on own drone"),
    ("RetakeSuspicious", "D3FEND Re-image (Restore)",        "retake a compromised drone"),
    ("RetakeRandom",     "D3FEND Re-image (Restore)",        "retake a random drone"),
    ("BlockSuspicious",  "D3FEND Network Isolation/M1037",   "block a compromised drone"),
    ("AllowTraffic",     "connectivity restore",             "re-allow blocked traffic"),
    ("DeployDecoy",      "D3FEND Decoy Service (deception)",  "plant a honeypot/decoy (CC2-winning tactic)"),
    ("DetectJam",        "D3FEND Signal/Anomaly Detection",  "[passive] SNR-threshold jam detector"),
    ("DetectGPS",        "D3FEND Sensor Cross-Validation",   "[passive] IMU cross-check GPS detector"),
    ("SafeMode",         "D3FEND Restore (position)",        "[passive] correct spoofed position"),
    ("Isolate",          "D3FEND Network Isolation",         "[passive] isolate flagged entity"),
]

RED_N = len(RED_CATALOG)                 # red decisions
BLUE_DECISION_N = 9                      # ids 0-8 are per-step choices


# ----------------------------------------------------------- obs helpers ---
def own_ip(obs, name):
    key = "drone_" + name.split("_")[-1]
    e = obs.get(key)
    if e and "Interface" in e:
        return e["Interface"][0]["IP Address"]
    return None


def ip_list(obs):
    return [v["Interface"][0]["IP Address"] for k, v in obs.items()
            if k != "success" and isinstance(v, dict) and "Interface" in v]


def _positions(obs):
    out = {}
    for k, v in obs.items():
        if k == "success" or not isinstance(v, dict):
            continue
        if "Interface" in v and "System info" in v and "position" in v["System info"]:
            out[v["Interface"][0]["IP Address"]] = v["System info"]["position"]
    return out


def _pick_by_distance(obs, name, farthest):
    """Return the nearest/farthest visible drone ip from own position."""
    pos = _positions(obs)
    me = own_ip(obs, name)
    if me not in pos or len(pos) < 2:
        ips = ip_list(obs)
        return ips[0] if ips else None
    ox, oy = pos[me]
    best, best_d = None, (-1.0 if farthest else 1e18)
    for ip, (x, y) in pos.items():
        if ip == me:
            continue
        d = sqrt((ox - x) ** 2 + (oy - y) ** 2)
        if (d > best_d) if farthest else (d < best_d):
            best, best_d = ip, d
    return best


# --------------------------------------------------------- red -> sim ---
def make_red_action(aid, obs, name, mem, np_random):
    """Build a concrete sim Action for red catalog id `aid`.

    `mem` is a per-agent dict; we use mem['target'] to sequence Exploit->Seize
    exactly like RedDroneWorm. Returns (action_obj, effective_aid).
    """
    ips = ip_list(obs)
    me = own_ip(obs, name)
    targets = [ip for ip in ips if ip != me] or ips

    def exploit(ip):
        mem["target"] = ip
        return ExploitDroneVulnerability(ip_address=ip, agent=name, session=0)

    if aid == 0 or aid == 1:                                  # Sleep / Discover
        return Sleep(), aid
    if aid == 2 and targets:                                  # ExploitNearest
        return exploit(_pick_by_distance(obs, name, farthest=False)), 2
    if aid == 3 and targets:                                  # ExploitRandom
        return exploit(np_random.choice(targets)), 3
    if aid == 4 and targets:                                  # ExploitFarthest
        return exploit(_pick_by_distance(obs, name, farthest=True)), 4
    if aid == 5:                                              # SeizeControl (+ persist hold)
        tip = mem.pop("target", None)
        if tip is not None:
            mem["held"] = tip
            return SeizeControl(ip_address=tip, agent=name, session=0), 5
        return Sleep(), 0
    if aid == 6 and targets:                                  # SpreadWorm (lateral)
        return exploit(_pick_by_distance(obs, name, farthest=False)), 6
    if aid == 7 and targets:                                  # JamNearest
        return FloodBandwidth(ip_address=_pick_by_distance(obs, name, False),
                              agent=name, session=0), 7
    if aid == 8 and targets:                                  # JamFarthest
        return FloodBandwidth(ip_address=_pick_by_distance(obs, name, True),
                              agent=name, session=0), 8
    if aid == 9 and targets:                                  # BlockComms
        return BlockTraffic(ip_address=np_random.choice(targets),
                            agent=name, session=0), 9
    if aid == 10:                                             # Persist (re-assert control)
        held = mem.get("held")
        if held is not None:
            return SeizeControl(ip_address=held, agent=name, session=0), 10
        return Sleep(), 0
    return Sleep(), 0                                         # fallback


# -------------------------------------------------------- blue -> index ---
def action_index_map(env, agent):
    """{action class name -> [(wrapper_index, ip), ...]} for one blue agent."""
    out = {}
    for i, action in env.agent_actions[agent].items():
        out.setdefault(type(action).__name__, []).append(
            (i, getattr(action, "ip_address", None)))
    return out


def make_blue_index(aid, env, agent, ctx):
    """Map blue catalog id `aid` -> a wrapper action index.

    ctx = {compromised: set[int], ip_to_drone: dict[ip->id]}.
    Observe-only decisions (Sleep/Monitor/Analyse) resolve to the Sleep index.
    """
    idx = action_index_map(env, agent)
    sleep = idx.get("Sleep", [(0, None)])[0][0]
    comp, ip2d = ctx["compromised"], ctx["ip_to_drone"]

    if aid in (0, 1, 2):                                      # Sleep/Monitor/Analyse
        return sleep
    if aid == 3:                                              # RemoveSessions
        return idx.get("RemoveOtherSessions", [(sleep, None)])[0][0]
    if aid == 4:                                              # RetakeSuspicious
        for i, ip in idx.get("RetakeControl", []):
            if ip2d.get(ip) in comp:
                return i
        return idx.get("RetakeControl", [(sleep, None)])[0][0]
    if aid == 5:                                              # RetakeRandom
        return idx.get("RetakeControl", [(sleep, None)])[0][0]
    if aid == 6:                                              # BlockSuspicious
        for i, ip in idx.get("BlockTraffic", []):
            if ip2d.get(ip) in comp:
                return i
        return idx.get("BlockTraffic", [(sleep, None)])[0][0]
    if aid == 7:                                              # AllowTraffic
        return idx.get("AllowTraffic", [(sleep, None)])[0][0]
    if aid == 8:                                              # DeployDecoy (deception; no CC3 primitive -> observe)
        return sleep
    return sleep


# ----------------------------------------------------------- LLM menus ---
def red_menu_text():
    return "\n".join(f"{i}. {n} [{m}] - {d}"
                     for i, (n, m, d) in enumerate(RED_CATALOG))


def blue_menu_text():
    return "\n".join(f"{i}. {n} [{m}] - {d}"
                     for i, (n, m, d) in enumerate(BLUE_CATALOG[:BLUE_DECISION_N]))
