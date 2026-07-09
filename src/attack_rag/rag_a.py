# RAG-A: 관측 → 공격타입 판단. 출력(B1) = [{id, name, confidence, target_host, attack_class}]
# 개선: ①attack_class 태그(감염/비감염/재밍/unknown) — HVT-vs-RAG 라우팅 + escalation 신호
#       ②abstention — 신뢰도 낮으면 unknown(추측 대신 안전기본)
import json, numpy as np
from sentence_transformers import SentenceTransformer

# --- attack_class 분류 사전 (ATT&CK tactics + 관측 layer + 키워드) ---
CONF_MIN = 0.35          # top 신뢰도 이하 = unknown(abstention)
_JAM_KW = ("jam", "denial of service", "gps", "spoof", "radio", "interference",
           "signal degrad", "navigation", "position drift", "satcom", "satellite",
           "frequency", "rf ", "link loss", "snr")
_JAM_IDS = {"T1464", "T1498", "T0814", "T1498.001", "T1498.002", "CAPEC-601", "CAPEC-603"}
_INTEG_KW = ("firmware", "supply chain", "supply-chain", "side channel", "side-channel",
             "adversarial", "evasion", "prompt", "poison", "tamper", "pre-os", "boot",
             "model inversion", "data manipulation", "integrity", "insider", "trojan")
_COMP_KW = ("lateral", "remote service", "exploit", "session", "worm", "botnet",
            "command and control", "c2", "takeover", "privilege", "execution",
            "backdoor", "implant", "injection", "hijack", "compromise")
_COMP_TAC = {"lateral-movement", "execution", "privilege-escalation",
             "command-and-control", "initial-access", "credential-access"}


import os
_HERE = os.path.dirname(os.path.abspath(__file__))


class RagA:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.kb = json.load(open(os.path.join(_HERE, 'attack_capec_kb.json'), encoding='utf-8'))
        self.xw = json.load(open(os.path.join(_HERE, 'drone_crosswalk.json'), encoding='utf-8'))
        self.ids = list(self.kb.keys())
        self.m = SentenceTransformer(model)
        corpus = [f"{self.kb[i]['name']}. {self.kb[i]['description']} {self.kb[i].get('detection','')}" for i in self.ids]
        self.E = self.m.encode(corpus, normalize_embeddings=True, batch_size=64)
        self.xwE = self.m.encode([x['itext'] for x in self.xw], normalize_embeddings=True)

    def _classify(self, tech, matched):
        """공격 클래스 유도: 재밍 / 비감염(무결성) / 감염 / (호출부에서 unknown)."""
        text = f"{tech.get('name','')} {tech.get('description','')} {tech.get('tactics','')}".lower()
        tacs = str(tech.get('tactics', '')).lower()
        tid = tech.get('id', '')
        layer = ""
        if matched:
            text += " " + matched.get('itext', '').lower() + " " + matched.get('signal', '').lower()
            layer = matched.get('layer', '')
        # ① 재밍/거부: RF 계층 or 재밍 키워드 or 재밍 기법ID
        if "물리/RF" in layer or tid in _JAM_IDS or any(k in text for k in _JAM_KW):
            # 단, 무결성/감염 키워드가 더 강하면 아래로 양보(예: GPS 언급뿐인 감염)
            if not any(k in text for k in ("worm", "lateral", "firmware", "side channel")):
                return "jamming"
        # ② 비감염/무결성 (사이드채널·적대적ML·펌웨어·인지공격)
        if any(k in text for k in _INTEG_KW):
            return "non_compromise"
        # ③ 감염/횡적이동
        if any(k in text for k in _COMP_KW) or any(t in tacs for t in _COMP_TAC):
            return "compromise"
        # 기본: 활성 탐지된 위협 → 봉쇄 우선(NIST 컨테인먼트-우선) = compromise 취급
        return "compromise"

    def identify(self, obs_text, target_host=None, topk=5):
        """obs_text: 관측 raw 설명(한/영). target_host: 공격받는 드론 id(들)."""
        q = self.m.encode([obs_text], normalize_embeddings=True)
        xw_sim = (q @ self.xwE.T).ravel(); best = int(xw_sim.argmax())
        matched = self.xw[best] if xw_sim[best] > 0.35 else None
        qtext = obs_text + " " + matched['itext'] if matched else obs_text
        qe = self.m.encode([qtext], normalize_embeddings=True)
        sim = (qe @ self.E.T).ravel(); top = sim.argsort()[::-1][:topk]
        out = []
        for j in top:
            k = self.kb[self.ids[j]]
            out.append({"id": k['id'], "name": k['name'], "domain": k['domain'],
                        "confidence": round(float(sim[j]), 3), "target_host": target_host,
                        "attack_class": self._classify(k, matched)})
        top_conf = out[0]['confidence'] if out else 0.0
        abstain = top_conf < CONF_MIN
        # aggregate class (NIST 봉쇄우선: top-3 어디든 감염 징후면 → compromise 에스컬레이션).
        top3 = [o['attack_class'] for o in out[:3]]
        if abstain:
            attack_class = "unknown"           # 신뢰도 미달 → 보수적 봉쇄로 라우팅(NIST 기본)
        elif "compromise" in top3:
            attack_class = "compromise"        # 감염 징후 존재 → 봉쇄우선(혼합공격 포함)
        elif "non_compromise" in top3[:2]:
            attack_class = "non_compromise"    # 무결성 변조 상위 → 봉쇄우선(tamper)
        else:
            attack_class = out[0]['attack_class']   # 순수 재밍/거부
        return {"result": out,
                "attack_class": attack_class,          # ★상위 라우팅 신호
                "abstain": abstain,                    # ★신뢰도 미달 → 안전기본
                "matched_signal": matched['signal'] if matched else None,
                "fallback": matched is None}


if __name__ == "__main__":
    r = RagA()
    for obs, host in [("드론 3번 신호세기 SNR 급락하고 통신 두절, 인접 드론도 하락", [3, 5]),
                      ("GPS 위치가 계속 표류하며 고도 이상, 신호세기는 정상", [7]),
                      ("인접 드론들이 순차적으로 감염되며 확산 중", [2, 4, 6]),
                      ("maintenance technician modifies firmware and mission waypoints", ["fleet"]),
                      ("", [])]:
        o = r.identify(obs, host)
        print(f"\n[관측] {obs[:45]}")
        print(f"  class={o['attack_class']} abstain={o['abstain']} 매칭={o['matched_signal']} fallback={o['fallback']}")
        for x in o['result'][:2]:
            print(f"    {x['id']:10} {x['name'][:34]:34} conf={x['confidence']} class={x['attack_class']}")
