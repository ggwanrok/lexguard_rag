# app/utils/diff.py
import difflib
from typing import List, Dict

def _trim_ws(text: str, i: int, j: int):
    while i < j and text[i].isspace(): i += 1
    while j > i and text[j-1].isspace(): j -= 1
    return i, j

def compute_revised_spans(original: str, revised: str) -> List[Dict]:
    """
    입력은 이미 특수문자 처리된 문자열이라고 가정.
    revised 기준 insert/replace만 하이라이트 대상으로 산출.
    """
    if not original or not revised or original == revised:
        return []
    sm = difflib.SequenceMatcher(a=original, b=revised, autojunk=False)
    spans: List[Dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        j1, j2 = _trim_ws(revised, j1, j2)
        if j2 <= j1:
            continue
        spans.append({"kind": "insert" if tag == "insert" else "replace",
                      "start": j1, "end": j2})
    if not spans:
        return []
    spans.sort(key=lambda x: x["start"])
    merged = [spans[0]]
    for sp in spans[1:]:
        last = merged[-1]
        if sp["start"] <= last["end"]:
            last["end"] = max(last["end"], sp["end"])
            if last["kind"] != sp["kind"]:
                last["kind"] = "replace"
        else:
            merged.append(sp)
    for m in merged:
        m["text"] = revised[m["start"]:m["end"]]
    return merged
