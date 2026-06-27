"""LLM action-selection backend for the 'llm' agent type.

Default is a fully offline, deterministic stub so the whole 3x3 sweep runs with
no network and is reproducible. If ANTHROPIC_API_KEY is set, decisions are made
by Claude instead, falling back to the stub on any error so a run never breaks.

The caller (brains.py) supplies `stub_fn` -- the offline decision -- so the stub
can use structured context while the prompt stays human-readable for the API.
"""
import os
import re

MODEL = os.environ.get("LAB_LLM_MODEL", "claude-haiku-4-5-20251001")

_client = None          # lazily created anthropic client
_warned = False


def available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _ask_claude(prompt, valid_ids):
    msg = _get_client().messages.create(
        model=MODEL, max_tokens=16, temperature=0.0,
        system=("You are a cyber operations decision agent. Choose exactly one "
                "action by replying with ONLY its integer id, nothing else."),
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    for n in nums:
        if n in valid_ids:
            return n
    raise ValueError(f"no valid id in LLM reply {text!r}")


def choose(prompt, valid_ids, stub_fn):
    """Return an action id in `valid_ids`. Uses Claude if a key is set, else stub."""
    global _warned
    if available():
        try:
            return _ask_claude(prompt, list(valid_ids))
        except Exception as e:                       # noqa: BLE001 - never break a run
            if not _warned:
                print(f"  [llm] API unavailable ({e}); using offline stub")
                _warned = True
    return stub_fn()
