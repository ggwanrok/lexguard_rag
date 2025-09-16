from pathlib import Path
import json, yaml
from typing import List

class ContractTypePack:
    def __init__(self, root: Path):
        self.root = root

        # 기본 리소스
        self.taxonomy = json.loads((root / "taxonomy.json").read_text(encoding="utf-8"))
        self.checklists = yaml.safe_load((root / "checklists.yaml").read_text(encoding="utf-8")) or {}
        self.router = yaml.safe_load((root / "router.yaml").read_text(encoding="utf-8")) or {}

        # ── 프롬프트 로딩 (감소치리: 없으면 빈 문자열 반환 → 호출부에서 무시)
        self.prompts = {
            "clause_audit": self._read_prompt("clause_audit.j2"),
            "clause_rewriter": self._read_prompt("clause_rewriter.j2"),  # NEW
        }

    def _read_prompt(self, filename: str) -> str:
        p = self.root / "prompts" / filename
        return p.read_text(encoding="utf-8") if p.exists() else ""

    @property
    def version(self) -> str:
        return self.taxonomy.get("version", "0")

    def categories(self) -> List[str]:
        # core + extra (중복 제거, 순서 보존)
        core = self.taxonomy.get("core", []) or []
        extra = self.taxonomy.get("extra", []) or []
        seen, ordered = set(), []
        for c in core + extra:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    def checklist(self, category: str) -> List[str]:
        return self.checklists.get(category, []) or []

    def route(self, role: str, jurisdiction: str) -> str:
        key = f"{(role or '').lower()}::{(jurisdiction or '').upper()}"
        return self.router.get(key, self.router.get("default", "default_v1"))

def load_pack(contract_type: str) -> ContractTypePack:
    base = Path(__file__).resolve().parent
    path = base / contract_type.lower()
    if not path.exists():
        raise FileNotFoundError(f"Pack not found for type: {contract_type} (expected at {path})")
    return ContractTypePack(path)
