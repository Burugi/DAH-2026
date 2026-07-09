"""A4: ATT&CK id -> D3FEND 방어 기법 직접 조회 (1차 경로).

ATT&CK id가 확정된 공격은 벡터 검색보다 공식 매핑 lookup이 정확하다.
attack_to_d3fend.json(build_kb.py 산출물)을 읽어 조회만 한다.

  AttackLookup().lookup("T1210") -> {"attack_label":..., "defenses":[...]}
  sub-technique(T1595.002)는 부모(T1595)로 폴백 조회.
"""
import json

from . import config


class AttackLookup:
    def __init__(self):
        with open(config.ATTACK_MAP_JSON) as f:
            self.map = json.load(f)

    def lookup(self, attack_id):
        """확정 ATT&CK id로 방어 기법 목록 조회. 없으면 None."""
        if not attack_id:
            return None
        hit = self.map.get(attack_id)
        if hit is None and "." in attack_id:        # sub-technique -> 부모 폴백
            hit = self.map.get(attack_id.split(".")[0])
        return hit

    def defenses(self, attack_id):
        """방어 기법 리스트만 반환 (없으면 빈 리스트)."""
        hit = self.lookup(attack_id)
        return hit["defenses"] if hit else []

    def covered(self, attack_id):
        return self.lookup(attack_id) is not None
