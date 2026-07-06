"""A6: 방어 RAG 파이프라인 골격 — 탐지 입력 -> 추천 blue 행동.

경로 선택(하이브리드):
  ① technique_id 확정 & 공식 매핑 존재 -> AttackLookup 직접 조회 (정확)
  ② id 미확정 / 매핑 없음(처음 보는 공격) -> observation 텍스트로 벡터 검색 (폴백)
두 경로 결과를 (d3fend_id, label, tactic)으로 정규화 -> action_map으로 blue 행동 변환.

입력/출력 형식은 DEFENSE_RAG_PLAN.md §6 인터페이스 초안을 따른다.
LLM은 선택적 근거 생성에만 쓰이며, 없으면 규칙 기반 근거로 폴백(오프라인 보장).
"""
import json

from . import action_map, config
from .index import Retriever
from .lookup import AttackLookup

# 추천 정렬 우선순위: 능동 봉쇄/제거를 관찰·강화보다 앞세운다.
# (관측만 잔뜩 나오면 방어 추천으로서 쓸모가 떨어지므로)
_TACTIC_PRIORITY = {"Isolate": 0, "Evict": 1, "Restore": 2, "Deceive": 3,
                    "Detect": 4, "Harden": 5, "Model": 6}


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

    # -- 한 탐지 처리 -------------------------------------------------------
    def recommend_one(self, det):
        tid = det.get("technique_id")
        target = det.get("target_host")
        obs = det.get("observation", "")

        if tid and self.lookup.covered(tid):
            path, techs = "direct_mapping", self._from_lookup(tid)
        else:
            path, techs = "vector_search", self._from_vector(obs)

        # 능동 대응(격리/제거)을 앞세우되 벡터 점수를 2차 기준으로 정렬
        techs.sort(key=lambda t: (_TACTIC_PRIORITY.get(t["tactic"], 9),
                                  -(t["score"] or 0.0)))

        # tactic 다양성을 살려 상위 몇 개만 blue 행동으로 변환
        seen_action, actions = set(), []
        for t in techs[: self.k]:
            aid, aname = action_map.d3fend_to_action(t["d3fend_id"], t["tactic"])
            if aid in seen_action:
                continue
            seen_action.add(aid)
            actions.append({
                "cyborg_action_id": aid,
                "cyborg_action": aname,
                "target": target,
                "d3fend_id": t["d3fend_id"],
                "d3fend_label": t["label"],
                "d3fend_tactic": t["tactic"],
                "score": t["score"],
                "rationale": self._rationale(det, t, aname, path),
            })
        return {
            "technique_id": tid, "path": path, "target": target,
            "n_candidates": len(techs), "recommended_actions": actions,
        }

    def recommend(self, detection_input):
        return {"results": [self.recommend_one(d)
                            for d in detection_input.get("detections", [])]}

    # -- 근거 생성 (규칙 기반 폴백; LLM 훅은 별도 확장) ---------------------
    def _rationale(self, det, tech, action, path):
        if path == "direct_mapping":
            basis = f"ATT&CK {det.get('technique_id')} 공식 매핑상 D3FEND {tech['label']}"
        else:
            basis = (f"관측('{(det.get('observation') or '')[:40]}...')과 "
                     f"유사도 {tech['score']}로 D3FEND {tech['label']}")
        return f"{basis}({tech['tactic']}) 권고 -> blue 행동 '{action}' 채택"


if __name__ == "__main__":
    rag = DefenseRAG()
    mock = {"detections": [
        {"technique_id": "T1210", "confidence": 0.9,
         "observation": "remote service exploited on drone", "target_host": "drone_07"},
        {"technique_id": None, "confidence": 0.4,
         "observation": "many hosts probed for open ports, scanning traffic",
         "target_host": "drone_03"},
        {"technique_id": "T1499", "confidence": 0.7,   # 매핑 없음 -> 벡터 폴백
         "observation": "sustained bandwidth flooding, comms availability degraded",
         "target_host": "drone_11"},
    ]}
    print(json.dumps(rag.recommend(mock), ensure_ascii=False, indent=2))
