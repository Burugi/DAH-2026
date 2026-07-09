"""A1+A2: D3FEND 원본 덤프 → 지식 베이스(KB) 파싱.

입력 (data/ 에 미리 받아둠, download_data.sh 참고):
  d3fend.json                 D3FEND 전체 온톨로지 (JSON-LD, @graph 7000+ 노드)
  d3fend-full-mappings.json   ATT&CK <-> D3FEND 공식 추론 매핑 (SPARQL 결과)

출력:
  d3fend_techniques.jsonl     방어 기법 271개, 1기법=1문서 (벡터 DB 원료)
  attack_to_d3fend.json       {ATT&CK id -> [방어 기법...]} 1차 조회용 lookup 테이블

이 스크립트는 순수 파싱만 한다(네트워크/임베딩 없음). 언제든 재실행 가능.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# D3FEND 7대 방어 전술(tactic). 기법은 이 중 하나에 속한다.
TACTICS = {"d3f:Model", "d3f:Harden", "d3f:Detect", "d3f:Isolate",
           "d3f:Deceive", "d3f:Evict", "d3f:Restore"}


def _load_graph():
    with open(os.path.join(DATA, "d3fend.json")) as f:
        g = json.load(f)["@graph"]
    return g, {n["@id"]: n for n in g if "@id" in n}


def _top_tactic(nid, byid, seen=None):
    """subClassOf 체인을 타고 올라가 기법이 속한 최상위 방어 전술을 찾는다."""
    seen = seen or set()
    n = byid.get(nid)
    if not n or nid in seen:
        return None
    seen.add(nid)
    e = n.get("d3f:enables")
    if e:
        eid = e["@id"] if isinstance(e, dict) else e
        if eid in TACTICS:
            return eid
    sc = n.get("rdfs:subClassOf")
    if not sc:
        return None
    if isinstance(sc, dict):
        sc = [sc]
    for p in sc:
        pid = p.get("@id") if isinstance(p, dict) else p
        if not pid or pid.startswith("_:"):
            continue
        r = _top_tactic(pid, byid, seen)
        if r:
            return r
    return None


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def build_techniques(g, byid):
    """D3- 방어 기법 271개를 문서 형태로 정제."""
    docs = []
    for n in g:
        did = str(n.get("d3f:d3fend-id", ""))
        if not did.startswith("D3-"):
            continue
        tactic = (_top_tactic(n["@id"], byid) or "d3f:Unknown").split(":")[-1]
        synonyms = [s for s in _as_list(n.get("d3f:synonym")) if isinstance(s, str)]
        docs.append({
            "d3fend_id": did,
            "label": n.get("rdfs:label", ""),
            "tactic": tactic,                       # Detect / Harden / Isolate ...
            "definition": n.get("d3f:definition", ""),
            "synonyms": synonyms,
            "uri": n["@id"],
        })
    docs.sort(key=lambda d: d["d3fend_id"])
    return docs


def build_mapping():
    """ATT&CK id -> 그 공격을 막는 D3FEND 기법 목록. 공식 추론 매핑에서 축약."""
    with open(os.path.join(DATA, "d3fend-full-mappings.json")) as f:
        bindings = json.load(f)["results"]["bindings"]

    def val(row, key):
        return row.get(key, {}).get("value")

    mapping = {}
    for r in bindings:
        oid = val(r, "off_tech_id")
        if not oid:
            continue
        def_tech = val(r, "def_tech_label")
        if not def_tech:
            continue
        entry = mapping.setdefault(oid, {
            "attack_label": val(r, "off_tech_label"),
            "defenses": {},          # def_tech_label -> 상세 (중복 제거용 dict)
        })
        entry["defenses"].setdefault(def_tech, {
            "def_tech": def_tech,
            "top_def_tech": val(r, "top_def_tech_label"),
            "def_tactic": val(r, "def_tactic_label"),
        })
    # dict -> list 로 정리
    for oid, e in mapping.items():
        e["defenses"] = sorted(e["defenses"].values(), key=lambda d: d["def_tech"])
    return mapping


def main():
    g, byid = _load_graph()
    techs = build_techniques(g, byid)
    mapping = build_mapping()

    out_t = os.path.join(HERE, "d3fend_techniques.jsonl")
    with open(out_t, "w") as f:
        for d in techs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    out_m = os.path.join(HERE, "attack_to_d3fend.json")
    with open(out_m, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)

    import collections
    tac = collections.Counter(d["tactic"] for d in techs)
    print(f"[techniques] {len(techs)}개 -> {out_t}")
    print(f"             tactic 분포: {dict(tac)}")
    print(f"[mapping]    ATT&CK {len(mapping)}개 -> {out_m}")
    total_pairs = sum(len(e['defenses']) for e in mapping.values())
    print(f"             (ATT&CK, D3FEND) 쌍: {total_pairs}")


if __name__ == "__main__":
    main()
