"""LLM 근거 생성 훅 — 후보 방어 행동 중 상황에 맞는 것 선택 + 자연어 근거.

기본은 완전 오프라인(규칙 기반). ANTHROPIC_API_KEY가 있으면 Claude가
후보 중 하나를 고르고 근거를 쓴다. 어떤 오류든 규칙 기반으로 폴백하므로
시연 중 네트워크가 없어도 파이프라인이 절대 깨지지 않는다.
(agents/llm.py의 오프라인-우선 패턴과 동일한 계약.)

모델: env DEFENSE_RAG_LLM_MODEL (기본 claude-sonnet-5, 계획서 §4).
구조화 출력(output_config.format)으로 {action_id, rationale}만 받아 파싱 안정성 확보.
sonnet-5/opus-4.8은 temperature/top_p를 거부하므로 넘기지 않는다.
"""
import json
import os

MODEL = os.environ.get("DEFENSE_RAG_LLM_MODEL", "claude-sonnet-5")

_client = None
_warned = False

_SYSTEM = (
    "당신은 드론 군집 사이버 방어 결정 에이전트다. 탐지된 공격과 후보 방어 행동"
    "(CybORG blue 행동)을 보고, 현재 상황에 가장 적합한 행동 하나를 고른다. "
    "반드시 주어진 후보의 action_id 중에서만 고르고, 근거는 한국어 한 문장으로 쓴다."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "action_id": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["action_id", "rationale"],
    "additionalProperties": False,
}


def available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _build_prompt(det, path, matched, actions):
    tid = matched or det.get("technique_id")
    lines = [
        f"공격 탐지: technique_id={tid}, 경로={path}",
        f"관측: {det.get('observation') or '(없음)'}",
        f"대상: {det.get('target')}",
        "",
        "후보 방어 행동:",
    ]
    for a in actions:
        sim = f", 유사도={a['score']}" if a.get("score") is not None else ""
        lines.append(
            f"  action_id={a['action_id']} {a['action_name']} "
            f"(D3FEND {a['d3fend_technique']} / {a['d3fend_tactic']}{sim})")
    lines.append("")
    lines.append("가장 적합한 action_id 하나와 한국어 근거 한 문장을 답하라.")
    return "\n".join(lines)


def _ask_claude(det, path, matched, actions):
    valid = {a["action_id"] for a in actions}
    resp = _get_client().messages.create(
        model=MODEL, max_tokens=1024,
        system=_SYSTEM,
        output_config={"effort": "low",
                       "format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user",
                   "content": _build_prompt(det, path, matched, actions)}])
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)                       # 스키마 강제라 파싱 안전
    aid = int(data["action_id"])
    if aid not in valid:                          # 환각 방지: 후보 밖이면 폴백
        raise ValueError(f"LLM이 후보 밖 action_id={aid} 선택")
    return aid, str(data["rationale"])


def select(det, path, matched, actions):
    """후보 actions 중 LLM이 고른 것을 맨 앞으로, 근거 교체.

    반환: (재정렬된 actions, llm_used). 키 없음/오류 시 원본 그대로 + False.
    """
    global _warned
    if not actions or not available():
        return actions, False
    try:
        aid, rationale = _ask_claude(det, path, matched, actions)
        chosen = next(a for a in actions if a["action_id"] == aid)  # 방어적: 밖이면 StopIteration
    except Exception as e:                         # noqa: BLE001 - 절대 파이프라인 중단 금지
        if not _warned:
            print(f"  [defense_rag.llm] API 불가({e}); 규칙 기반 근거 사용")
            _warned = True
        return actions, False
    chosen = dict(chosen, rationale=rationale, llm_selected=True)
    rest = [a for a in actions if a["action_id"] != aid]
    return [chosen] + rest, True
