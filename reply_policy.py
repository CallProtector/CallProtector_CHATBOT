# reply_policy.py
from __future__ import annotations
import re, json
from typing import List, Dict, Tuple

# ---- Smalltalk --------------------------------------------------------------
SMALLTALK_KWS = [
    "안녕","안뇽","하이","hi","hello","헬로","헤이","방가","ㅎㅇ","그냥",
    "잘 지내","뭐해","심심해","심심","ㅎㅎ","ㅋㅋ","굿모닝","굿밤","잘자","좋은 아침","수고","고마워","땡큐","감사","thanks","thx","ㄳ",
    "테스트"
]
def is_smalltalk(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(k in t for k in SMALLTALK_KWS)

def smalltalk_reply(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["안녕","안뇽","하이","hello","hi","헬로","헤이","방가","ㅎㅇ"]):
        return "안녕하세요! 만나서 반가워요 😊 무엇을 도와드릴까요?"
    if any(k in t for k in ["굿모닝","좋은 아침"]):
        return "안녕하세요! 잘 지내셨나요? 😊 무엇을 도와드릴까요?"
    if any(k in t for k in ["굿밤","잘자"]):
        return "고마워요! 편안한 밤 되세요 🌛"
    if any(k in t for k in ["고마워","감사","땡큐","thx","thanks","수고","ㄳ"]):
        return "별말씀을요! 도움이 되어 기뻐요. 또 궁금한 점 있으면 편하게 물어보세요."
    if any(k in t for k in ["뭐해","심심해","심심"]):
        return "여기 있어요! 질문을 기다리는 중이에요. 어떤 도움이 필요하신가요?"
    if any(k in t for k in ["ㅎㅎ","ㅋㅋ","그냥"]):
        return "헤헤 😄 농담도 좋아요. 이제 본론으로—무엇을 도와드릴까요?"
    if "테스트": return "개발하느라 고생이 많아요. 그래도 끝까지 파이팅!💪"
    return "안녕하세요! 편하게 말씀해 주세요. 민원/상담 관련도 좋고, 일반적인 질문도 환영해요."
  

# ---- 키워드 → (유형, 법률) 1차 힌트 ----------------------------------------
def keyword_pairs_first(text: str, limit:int=5) -> List[Dict[str,str]]:
    hay = (text or "")
    out: List[Dict[str,str]] = []
    def add(u,l): out.append({"유형":u,"관련법률":l})
    if any(k in hay for k in ["성희롱","음란","음담"]):
        add("성희롱/음란발언","성폭력범죄의 처벌 등에 관한 특례법 제13조")
    if any(k in hay for k in ["욕설","협박","폭언"]):
        add("협박/폭행(폭언) 가능성","형법 제283조; 형법 제260조")
    if any(k in hay for k in ["모욕","명예훼손","폭언"]):
        add("명예훼손·모욕·폭언","형법 제307조; 형법 제311조")
    if "업무방해" in hay:
        add("업무방해","형법 제314조")
    if "강요" in hay:
        add("강요","형법 제324조")
    if any(k in hay for k in ["장난전화","괴롭힘"]):
        add("장난전화/경범","경범죄처벌법 제3조 제1항 제40호")
    if any(k in hay for k in ["반복적인 민원"]):
        add("반복(고질.강성민원)","경범죄처벌법 제3조 제1항 제40호")
    if "스토킹" in hay:
        add("스토킹","스토킹범죄의 처벌 등에 관한 법률 제18조")
    return out[:limit]
  
# ---- 법률 요약 사전 ---------------------------------------------------------
_LAW_BRIEFS = {
    "성폭력범죄의 처벌 등에 관한 특례법 제13조": "통신수단을 이용한 성적 수치심 유발 행위를 처벌합니다. (2년 이하 징역 또는 2천만원 이하 벌금)",
    "형법 제283조": "상대에게 공포심을 유발하는 협박 행위를 처벌합니다. (3년 이하 징역 또는 500만원 이하 벌금)",
    "형법 제260조": "상대방 신체에 대한 유형력 행사(폭행)를 처벌합니다. (2년 이하 징역 또는 500만원 이하 벌금)",
    "형법 제307조": "허위/사실 적시로 타인의 명예를 훼손하는 행위를 처벌합니다. (2년 이하 징역 또는 500만원 이하 벌금)",
    "형법 제311조": "공연한 모욕행위를 처벌합니다. (1년 이하 징역 또는 200만원 이하 벌금)",
    "형법 제314조": "위력 기타 방법으로 타인의 업무를 방해하는 행위를 처벌합니다. (5년 이하 징역 또는 1천5백만원 이하 벌금)",
    "형법 제324조": "폭행/협박 등으로 의사에 반해 의무 없는 일을 하게 하는 강요를 처벌합니다. (5년 이하 징역 또는 3천만원 이하 벌금)",
    "경범죄처벌법 제3조 제1항 제40호": "정당한 이유 없는 반복 전화 등 괴롭힘을 제재합니다. (10만원 이하 벌금·구류·과료)",
    "스토킹범죄의 처벌 등에 관한 법률 제18조": "지속·반복적 스토킹 범죄를 처벌하고 보호조치를 규정합니다. (3년 이하 징역 또는 3천만원 이하 벌금)",
    "국민권익위원회 상담사 보호 지침": "상담 과정에서 발생하는 욕설·폭언·성희롱 등 악·강성 민원으로부터 상담사를 보호하기 위해 마련된 제도적 지침입니다. 상담 종료 기준, 기록 관리, 보호 조치 절차 등을 규정합니다.",
}

def _brief_fallback_by_keyword(law: str) -> str:
    low = (law or "").lower()
    if "협박" in low: return "협박 행위 전반을 처벌합니다."
    if "폭행" in low: return "타인에 대한 유형력 행사(폭행)를 처벌합니다."
    if "모욕" in low: return "공연한 모욕을 처벌합니다."
    if "명예훼손" in low: return "허위/사실 적시 명예훼손을 처벌합니다."
    if "업무방해" in low: return "업무 수행을 방해하는 행위를 처벌합니다."
    if "스토킹" in low: return "지속·반복적 스토킹을 처벌하고 피해자 보호를 규정합니다."
    if "성폭력" in low or "이용음란" in low or "통신" in low: return "통신수단을 이용한 성적 수치심 유발 행위를 처벌합니다."
    return "관련 행위를 규율·제재하여 피해 방지를 도모합니다."

def brief_for_law(law: str) -> str:
    return _LAW_BRIEFS.get(law, _brief_fallback_by_keyword(law))
  
# ---- 정규화/후처리/포맷 ------------------------------------------------------
def normalize_law_name(law: str) -> str:
    if not law: return ""
    return re.sub(r"\s*\(.*?\)", "", law).strip()

def _ok(v: str | None) -> bool:
    v = (v or "").strip()
    return bool(v) and v not in ("없음","정보없음")

def post_filter_sources(sources: List[Dict], limit:int=3) -> List[Dict]:
    """
    - ';' ',' 로 합쳐진 법률 분할
    - 법률명(정규화) 기준 dedup (유형 달라도 같은 법률이면 1개)
    - '없음' 제거, 최대 limit 유지
    """
    out, seen = [], set()
    for e in sources or []:
        typ = (e.get("유형") or "").strip()
        raw = (e.get("관련법률") or "").strip()
        if not (_ok(typ) and _ok(raw)): 
            continue
        parts = [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]
        for lw in parts:
            norm = normalize_law_name(lw)
            if not _ok(norm): 
                continue
            key = norm.lower()
            if key in seen: 
                continue
            seen.add(key)
            out.append({"유형": typ, "관련법률": norm})
            if len(out) >= limit:
                return out
    return out

def format_sourcepages_text(sources: List[Dict]) -> str:
    blocks = []
    for e in sources or []:
        t = (e.get("유형") or "").strip()
        l = (e.get("관련법률") or "").strip()
        if not t or not l: 
            continue
        blocks.append(f"- 유형: {t}\n- 관련법률: {l}")
    return "\n\n".join(blocks)



# ---- 2문단 보정 -------------------------------------------------------------
def build_second_paragraph(sources: List[Dict]) -> str:
    if not sources:
        return ("당신이 상담한 내용은 **‘해당 유형’**에 해당할 수 있으며, 관련 법률로는 **‘해당 법률’**이 있습니다.\n"
                "각 법률의 적용은 상황에 따라 달라질 수 있으니 기관 지침과 법률 자문을 함께 참고하세요.")
    typ = (sources[0].get("유형") or "해당 유형").strip()
    laws, seen = [], set()
    for e in sources:
        l = (e.get("관련법률") or "").strip()
        if not l or l in seen:
            continue
        seen.add(l)
        laws.append(l)
    head = f"당신이 상담한 내용은 **‘{typ}’**에 해당할 수 있으며, 관련 법률로는 **‘" + "’, ‘".join(laws) + "’**가 있습니다."
    # 법률명만 굵게
    bullets = "\n".join([f"- **{l}**: {brief_for_law(l)}" for l in laws])
    return head + "\n" + bullets

def ensure_two_paragraphs(answer: str, sources: List[Dict]) -> str:
    text = (answer or "").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = ["상황 기록, 증거 보존, 상급자 보고, 심리 안정 확보 등 즉시 조치를 진행하세요."]
    second = build_second_paragraph(sources)
    if len(paras) == 1:
        paras.append(second)
    else:
        paras[1] = second
    # 1문단 보강(4문장 미만이면 보충)
    sents = [s for s in re.split(r"[.。]\s*", paras[0]) if s.strip()]
    if len(sents) < 4:
        supplement = (" 통화 선종료·차단 기준을 숙지하고, 재발 방지를 위한 안내 멘트를 사용하세요. "
                      "내부 시스템에 시간/상황/발언을 구체 기록하고 즉시 보호조치를 요청하세요.")
        paras[0] = (paras[0] + supplement).strip()
    return "\n\n".join(paras)


# ---- 병합 우선순위 규칙(단일 진실 소스) ------------------------------------
# ‘키워드 → RAG → 모델’ 순으로 병합 (모델은 마지막: 환각 가능성 때문)
def merge_sources(keyword_src: List[Dict], rag_src: List[Dict], model_src: List[Dict], limit:int=3) -> List[Dict]:
    merged = (keyword_src or []) + (rag_src or []) + (model_src or [])
    return post_filter_sources(merged, limit=limit)

# ---- JSON 파싱 보정 ----------------------------------------------------------
def parse_model_json(full_text: str) -> Tuple[str, List[Dict]]:
    """
    모델이 JSON을 스트림 출력했을 때 최종 합치기 후 파싱.
    - 실패 시 (원문, []) 반환
    - 성공 시 (answer, sourcePages[정규화]) 반환
    """
    answer = full_text
    sources: List[Dict] = []
    try:
        parsed = json.loads(full_text)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("answer"), str):
                answer = parsed["answer"]
            sp = parsed.get("sourcePages")
            if isinstance(sp, list):
                tmp = []
                for e in sp:
                    if not isinstance(e, dict): 
                        continue
                    t = (e.get("유형") or "").strip()
                    l = normalize_law_name((e.get("관련법률") or "").strip())
                    if _ok(t) and _ok(l):
                        tmp.append({"유형": t, "관련법률": l})
                sources = tmp
    except Exception:
        pass
    return answer, sources