"""Scenario loader for attack scenario YAML files.

Usage:
    from scenarios import load_scenario, list_scenarios

    # Load one scenario and merge its attacks into an existing config
    cfg = yaml.safe_load(open("configs/sweep.yaml"))
    cfg = load_scenario("A1", cfg)

    # List all scenarios with physical sim support
    for meta in list_scenarios(sim_only=True):
        print(meta["id"], meta["name"])
"""
import os
import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))


def _all_files():
    return sorted(
        f for f in os.listdir(_DIR)
        if f.endswith(".yaml") and not f.startswith("_")
    )


def load_scenario(scenario_id: str, cfg: dict | None = None) -> dict:
    """Load a scenario YAML by id (e.g. 'A1' or 'A14').

    If cfg is provided, merges the scenario's attacks, defense and worm blocks
    into a copy of cfg and returns it. Otherwise returns the raw scenario dict.
    """
    target_id = scenario_id.upper().lstrip("A").zfill(1)  # normalise
    for fname in _all_files():
        path = os.path.join(_DIR, fname)
        data = yaml.safe_load(open(path, encoding="utf-8"))
        if str(data.get("id", "")).upper().lstrip("A") == target_id:
            if cfg is None:
                return data
            merged = dict(cfg)
            # Merge attacks (append, never replace)
            existing = list(merged.get("attacks") or [])
            merged["attacks"] = existing + list(data.get("attacks") or [])
            # Merge defense (scenario values take precedence)
            if data.get("defense"):
                merged["defense"] = {**(merged.get("defense") or {}), **data["defense"]}
            # Merge worm (버그 수정: 기존엔 병합 누락 → run._worm_step의 cfg.get("worm")가
            # None이라 A01/A09/A14/A17 등의 웜이 실행되지 않았음. pre_compromise는
            # run.py가 cfg["_scenario"]에서 읽으므로 아래 _scenario 병합으로 이미 정상.)
            if data.get("worm"):
                merged["worm"] = data["worm"]
            merged["_scenario"] = data
            return merged
    raise FileNotFoundError(f"No scenario file found for id '{scenario_id}'")


def list_scenarios(sim_only: bool = False) -> list[dict]:
    """Return metadata dicts for all (or sim-capable) scenarios."""
    out = []
    for fname in _all_files():
        data = yaml.safe_load(open(os.path.join(_DIR, fname), encoding="utf-8"))
        has_sim = bool(data.get("attacks"))
        if sim_only and not has_sim:
            continue
        out.append({
            "id": data.get("id"),
            "name": data.get("name"),
            "has_sim": has_sim,
            "mitre_attack": data.get("mitre_attack", []),
        })
    return out
