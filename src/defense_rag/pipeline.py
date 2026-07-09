"""A6: 방어 RAG 파이프라인 골격 — 탐지 입력 -> 추천 blue 행동.

경로 선택(하이브리드):
  ① technique_id 확정 & 공식 매핑 존재 -> AttackLookup 직접 조회 (정확)
  ② id 미확정 / 매핑 없음(처음 보는 공격) -> observation 텍스트로 벡터 검색 (폴백)
두 경로 결과를 (d3fend_id, label, tactic)으로 정규화 -> action_map으로 blue 행동 변환.

입력/출력 형식은 DEFENSE_RAG_PLAN.md §6 확정 스키마를 따른다.
  - 입력 target: {"type": "hosts"|"area", "value": [...]}
    · hosts = 특정 감염 드론 지목 → 제거/재탈환(3·4·5) 허용
    · area  = 재밍/GPS 광역 → 지목 호스트 없음 → 제거/재탈환 제외
  - technique_id: 단일 문자열 또는 top-k 목록 모두 허용 (커버되는 첫 id로 직접 조회)
LLM은 선택적 근거 생성에만 쓰이며, 없으면 규칙 기반 근거로 폴백(오프라인 보장).
"""
import json

from . import action_map, config, llm_hook
from .index import Retriever
from .lookup import AttackLookup

# 추천 정렬 우선순위: 능동 봉쇄/제거를 관찰·강화보다 앞세운다.
# (관측만 잔뜩 나오면 방어 추천으로서 쓸모가 떨어지므로)
_TACTIC_PRIORITY = {"Isolate": 0, "Evict": 1, "Restore": 2, "Deceive": 3,
                    "Detect": 4, "Harden": 5, "Model": 6}

# 특정 감염 호스트가 지목돼야만 의미 있는 blue 행동(제거/재탈환).
# target.type == "area"(재밍·GPS 광역)면 지목 호스트가 없으므로 제외한다.
_PER_HOST_ACTIONS = {3, 4, 5}   # RemoveSessions / RetakeSuspicious / RetakeRandom


class DefenseRAG:
    def __init__(self, k=5):
        self.lookup = AttackLookup()
        self.retriever = Retriever()
        self.k = k
        # def_tech 라벨 -> 기법 문서(d3fend_id, tactic) 정합용 레지스트리
        self._by_label = {}
        with open(config.TECHNIQUES_JSONL) as f:
            for line in f:
                d = json.loads(line)
                self._by_label[d["label"].lower()] = d

    # -- 경로 ①: ATT&CK 직접 조회 결과를 정규화 -----------------------------
    def _from_lookup(self, attack_id):
        techs = []
        for dfn in self.lookup.defenses(attack_id):
            label = dfn["def_tech"]
            reg = self._by_label.get(label.lower(), {})
            techs.append({
                "d3fend_id": reg.get("d3fend_id", ""),
                "label": label,
                "tactic": reg.get("tactic") or (dfn.get("def_tactic") or ""),
                "score": None,                     # 직접 매핑엔 유사도 없음
            })
        return techs

    # -- 경로 ②: 관측 텍스트 벡터 검색 -------------------------------------
    def _from_vector(self, observation):
        if not observation:
            return []
        return [{"d3fend_id": h["d3fend_id"], "label": h["label"],
                 "tactic": h["tactic"], "score": h["score"]}
                for h in self.retriever.search(observation, self.k)]

    # -- 입력 정규화 (§6 확정 스키마) --------------------------------------
    @staticmethod
    def _candidate_ids(det):
        """technique_id: 단일 문자열 또는 top-k 목록 둘 다 허용 -> id 리스트."""
        tid = det.get("technique_id")
        if tid is None:
            return []
        return [tid] if isinstance(tid, str) else list(tid)

    @staticmethod
    def _target_out(target):
        """target {type, value} -> (type, 출력용 target 값). 단일이면 문자열."""
        if isinstance(target, dict):
            ttype = target.get("type", "hosts")
            val = target.get("value") or []
        else:                                    # 구형/문자열 호환
            ttype, val = "hosts", ([target] if target else [])
        out = val[0] if len(val) == 1 else val
        return ttype, out

    # -- 한 탐지 처리 -------------------------------------------------------
    def recommend_one(self, det):
        cand_ids = self._candidate_ids(det)
        obs = det.get("observation", "")
        ttype, target_out = self._target_out(det.get("target"))

        # ① 확정 id 중 매핑 커버되는 첫 번째로 직접 조회
        matched = next((i for i in cand_ids if self.lookup.covered(i)), None)
        if matched:
            path, techs = "direct_mapping", self._from_lookup(matched)
        else:                                    # ② 관측 텍스트 벡터 폴백
            path, techs = "vector_search", self._from_vector(obs)

        # 능동 대응(격리/제거)을 앞세우되 벡터 점수를 2차 기준으로 정렬
        techs.sort(key=lambda t: (_TACTIC_PRIORITY.get(t["tactic"], 9),
                                  -(t["score"] or 0.0)))

        # tactic 다양성을 살려 상위 몇 개만 blue 행동으로 변환
        seen_action, actions = set(), []
        for t in techs[: self.k]:
            aid, aname = action_map.d3fend_to_action(t["d3fend_id"], t["tactic"], t["label"])
            # area(광역)면 특정 호스트 지목이 필요한 제거/재탈환은 건너뛴다
            if ttype == "area" and aid in _PER_HOST_ACTIONS:
                continue
            if aid in seen_action:
                continue
            seen_action.add(aid)
            actions.append({
                # §6 확정 출력 스키마 필드
                "action_id": aid,
                "action_name": aname,
                "target": target_out,
                "d3fend_technique": t["d3fend_id"],
                "rationale": self._rationale(det, t, aname, path, matched),
                # 진단용 부가 필드 (소비자는 무시 가능)
                "d3fend_label": t["label"],
                "d3fend_tactic": t["tactic"],
                "score": t["score"],
            })

        # LLM 훅: 키 있으면 후보 중 최적 선택 + 근거 교체, 없으면 규칙 순서 유지
        actions, llm_used = llm_hook.select(det, path, matched, actions)

        return {
            "technique_id": matched or cand_ids, "path": path,
            "target_type": ttype, "target": target_out, "llm_used": llm_used,
            "n_candidates": len(techs), "recommended_actions": actions,
        }

    def recommend(self, detection_input):
        return {"results": [self.recommend_one(d)
                            for d in detection_input.get("detections", [])]}

    # -- 근거 생성 (규칙 기반 폴백; LLM 훅은 별도 확장) ---------------------
    def _rationale(self, det, tech, action, path, matched=None):
        if path == "direct_mapping":
            basis = f"ATT&CK {matched} 공식 매핑상 D3FEND {tech['label']}"
        else:
            basis = (f"관측('{(det.get('observation') or '')[:40]}...')과 "
                     f"유사도 {tech['score']}로 D3FEND {tech['label']}")
        return f"{basis}({tech['tactic']}) 권고 -> blue 행동 '{action}' 채택"


if __name__ == "__main__":
    rag = DefenseRAG()
    # §6 확정 스키마 mock: target = {type, value}, technique_id 단일/목록 혼용
    mock = {"detections": [
        {"technique_id": "T1210", "confidence": 0.9,
         "observation": "remote service exploited on drone",
         "target": {"type": "hosts", "value": ["drone_07"]}},
        {"technique_id": None, "confidence": 0.4,      # 미확정 -> 벡터 폴백
         "observation": "many hosts probed for open ports, scanning traffic",
         "target": {"type": "hosts", "value": ["drone_03"]}},
        {"technique_id": ["T1499", "T1498"], "confidence": 0.7,  # 매핑 없음 목록 -> 벡터 폴백, area 분기
         "observation": "sustained bandwidth flooding jamming, comms availability degraded",
         "target": {"type": "area", "value": ["sector_north"]}},
    ]}
    print(json.dumps(rag.recommend(mock), ensure_ascii=False, indent=2))
