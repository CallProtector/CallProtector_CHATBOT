import os, json, asyncio, random, re
from collections import defaultdict, deque
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone

# 공통 정책
from reply_policy import (
    is_smalltalk, smalltalk_reply,
    keyword_pairs_first,
    parse_model_json,
    merge_sources,
    ensure_two_paragraphs,
    format_sourcepages_text,
)

load_dotenv()
router = APIRouter()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# 오레곤(us-west-2) 인덱스 사용
index_name = os.getenv("PINECONE_INDEX", "legal-guideline-usw2")
index = pc.Index(index_name)

# ---- in-memory session ----
MAX_TURNS = 8
session_memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS))
session_locks: Dict[str, asyncio.Lock] = defaultdict(lambda: asyncio.Lock())

# ---- request model ----
class StreamQuery(BaseModel):
    session_id: int
    question: str
    # STT 스크립트: [{ "speaker": "INBOUND|OUTBOUND", "text": "..." }, ...]
    context_scripts: Optional[List[Dict[str, str]]] = None

# ---- helpers ----
def ns_key(session_id: int) -> str:
    # CALL/CHAT 통합 네임스페이스
    return "call:" + str(session_id)

# ----- canned 응답 (옵션) -----
async def slow_emit_json(payload: dict, min_wait: float = 1.8, max_wait: float = 3.0):
    """RAG처럼 보이도록 최종 JSON 전송을 살짝 지연."""
    await asyncio.sleep(random.uniform(min_wait, max_wait))
    yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "data: [END]\n\n"

PATTERNS = {
    "ABUSE_DETAIL": re.compile(r"이\s*중\s*폭언\s*관련된\s*법률.*자세히", re.I),
    "HARASS_DETAIL": re.compile(r"이\s*중\s*성희롱\s*관련된\s*법률.*자세히", re.I),
}

def match_canned_tag(text: str) -> Optional[str]:
    t = (text or "")
    for tag, pat in PATTERNS.items():
        if pat.search(t):
            return tag
    return None

def build_canned_payload(tag: str) -> Optional[dict]:
    if tag == "ABUSE_DETAIL":
        src = [
            {"유형": "협박/폭행(폭언) 가능성", "관련법률": "형법 제283조"},
            {"유형": "협박/폭행(폭언) 가능성", "관련법률": "형법 제260조"},
            {"유형": "명예훼손·모욕·폭언",   "관련법률": "형법 제307조"},
        ]
        answer = (
            "폭언(욕설/협박)에 해당하는 발언은 **\"시발 빡치네\"**, **\"죽고 싶냐\"**, **\"죽여 버린다\"**가 있어요. \n\n"
            "해당 발언은 **‘협박/폭행(폭언) 가능성’**에 해당할 수 있으며, 관련 법률로는 **‘형법 제283조’**, **‘형법 제260조’**, **‘형법 제307조’**가 있습니다.\n"
            "- **형법 제283조**: 상대에게 공포심을 유발하는 협박 행위를 처벌합니다. (3년 이하 징역 또는 500만원 이하 벌금)\n"
            "- **형법 제260조**: 상대방 신체에 대한 유형력 행사(폭행)를 처벌합니다. (2년 이하 징역 또는 500만원 이하 벌금)\n"
            "- **형법 제307조**: 허위/사실 적시로 타인의 명예를 훼손하는 행위를 처벌합니다. (2년 이하 징역 또는 500만원 이하 벌금)\n\n"
            "즉시 취해야 할 조치는 폭언을 명확히 인지하고 이를 기록하여 상급자에게 보고하는 것입니다. 상담사는 감정적으로 대응하지 않도록 주의하고, 필요 시 동료 상담을 통해 심리적 안정을 취하세요. 반복 시 차단/선종료 기준에 따라 조치하세요.\n\n"
            "상담사님의 건강한 근무 환경을 응원합니다 :)\n\n"
        )
        return {"answer": answer, "sourcePages": src, "sourcePagesText": format_sourcepages_text(src)}

    if tag == "HARASS_DETAIL":
        src = [
            {"유형": "성희롱/음란발언", "관련법률": "성폭력범죄의 처벌 등에 관한 특례법 제13조"},
        ]
        answer = (
            "성희롱에 해당하는 발언은 **\"목소리가 야 하시네요\"**가 있어요. \n\n"
            "해당 발언은 **‘성희롱/음란발언’**에 해당할 수 있으며, 관련 법률로는 **‘성폭력범죄의 처벌 등에 관한 특례법 제13조’**가 있습니다.\n"
            "- **성폭력범죄의 처벌 등에 관한 특례법 제13조**: 통신수단을 이용한 성적 수치심 유발 행위를 처벌합니다. (2년 이하 징역 또는 2천만원 이하 벌금)\n\n"
            "즉시 취해야 할 조치는 중지 요청 및 기록, 상급자 보고이며, 재발 시 ARS 경고 후 통화 종료 기준을 적용하세요. 피해자의 심리 안정 지원도 병행하세요.\n\n"
            "상담사님의 건강한 근무 환경을 응원합니다 :)\n\n"
        )
        return {"answer": answer, "sourcePages": src, "sourcePagesText": format_sourcepages_text(src)}

    return None


# --- helpers (추가) ---
TYPE_KEYWORDS = {
    "성희롱/음란발언": ["성희롱","음란","성적","목소리", "야하", "야한","섹스","음담","음탕"],
    "협박/폭행(폭언) 가능성": ["죽여버린다","폭언","욕설","협박","죽여","패버린다","죽고 싶냐","시발", "씨발","ㅅㅂ","개새"],
    "명예훼손·모욕·폭언": ["모욕","명예훼손","비방","바보","멍청","병신","등신"],
    "업무방해": ["업무방해","업무 방해"],
    "강요": ["강요"],
    "스토킹": ["스토킹","지속 연락","반복 연락"],
}

def detect_allowed_types(text: str) -> set[str]:
    hay = (text or "").lower()
    out = set()
    for typ, kws in TYPE_KEYWORDS.items():
        if any(k.lower() in hay for k in kws):
            out.add(typ)
    return out

def filter_sources_by_types(sources: List[dict], allowed: set[str]) -> List[dict]:
    if not allowed:
        return sources
    return [s for s in (sources or []) if s.get("유형") in allowed]

def _gather_banned_keywords(allowed_types: Optional[set[str]]) -> List[str]:
    """허용유형에 속하지 않는 유형들의 키워드를 모두 금지어로 수집."""
    if not allowed_types:
        return []
    banned: List[str] = []
    for typ, kws in TYPE_KEYWORDS.items():
        if typ not in allowed_types:
            banned.extend(kws)
    # 길이가 긴 키워드를 먼저 치환하도록 정렬 (오염 최소화)
    banned = sorted(set(banned), key=len, reverse=True)
    return banned

def _preferred_replacement_term(allowed_types: Optional[set[str]]) -> str:
    if allowed_types:
        if "협박/폭행(폭언) 가능성" in allowed_types:
            return "폭언"
        if "명예훼손·모욕·폭언" in allowed_types:
            return "모욕/폭언"
    return "부적절한 발언"

def _replace_banned_keywords(line: str, banned_keywords: List[str], replacement: str) -> str:
    out = line
    for kw in banned_keywords:
        # 단순 포함 기준으로, 대소문자 무시
        out = re.sub(re.escape(kw), replacement, out, flags=re.IGNORECASE)
    return out

def sanitize_answer_by_allowed(answer: str, final_sources: List[dict], allowed_types: Optional[set[str]]) -> str:
    """허용 유형/법률만 남기고, 1문단 일반 서술에도 금지 키워드가 나오지 않도록 치환."""
    if not answer:
        return answer

    allowed_type_names = {s.get("유형", "") for s in (final_sources or []) if s.get("유형")}
    allowed_laws = {s.get("관련법률", "") for s in (final_sources or []) if s.get("관련법률")}

    # 금지 키워드 수집 및 기본 치환어
    banned_keywords = _gather_banned_keywords(allowed_types or allowed_type_names)
    replacement = _preferred_replacement_term(allowed_types or allowed_type_names)

    cleaned_lines: List[str] = []
    for raw in answer.splitlines():
        line = raw.rstrip()

        # 2문단의 법률 라인만 허용법률로 필터
        if line.strip().startswith("- **"):
            lawname = line.strip().replace("- **", "").split("**", 1)[0].strip()
            if allowed_laws and lawname not in allowed_laws:
                continue

        # 2문단 첫 문장(헤더) 유형 교정
        if "당신이 상담한 내용은" in line:
            if allowed_type_names:
                # 가장 우선 유형 하나를 명시
                primary_type = next(iter(allowed_type_names))
                # 기존 문장을 통째로 교체해 안전화
                line = f"당신이 상담한 내용은 ‘{primary_type}’에 해당할 수 있으며, 관련 법률로는 다음과 같습니다."
            else:
                # 허용유형을 전혀 찾지 못하면 문장 제거
                continue

        # 1문단/기타 라인에서 금지 키워드 치환 (성희롱 등의 단어가 섞여 나오는 것을 방지)
        if banned_keywords:
            line = _replace_banned_keywords(line, banned_keywords, replacement)

        cleaned_lines.append(line)

    # 연속 빈줄 정리
    out: List[str] = []
    for l in cleaned_lines:
        if not out or l.strip() or out[-1].strip():
            out.append(l)
    return "\n".join(out).strip()


# ✅ RAG 검색 (원문 그대로 회수) + 허용유형 필터
def retrieve_context(query: str, top_k: int = 5, allowed_types: Optional[set[str]] = None) -> Tuple[str, List[dict]]:
    emb = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    results = index.query(vector=emb, top_k=top_k, include_metadata=True, include_values=False)

    blocks: List[str] = []
    sources: List[dict] = []
    matches = getattr(results, "matches", None) or (results.get("matches", []) if isinstance(results, dict) else [])

    for m in matches:
        meta = getattr(m, "metadata", None) or (m.get("metadata", {}) if isinstance(m, dict) else {}) or {}
        typ = (meta.get("유형") or "정보없음").strip()
        law_raw = (meta.get("관련 법률") or meta.get("관련법률") or "정보없음").strip()

        # 허용유형이 지정되어 있으면 그 외는 스킵
        if allowed_types and typ not in allowed_types:
            continue

        blocks.append(
            f"📌 **유형**: {typ or '정보없음'}\n"
            f"📖 본문: {meta.get('본문','')}\n"
            f"⚖ **관련 법률**: {law_raw or '정보없음'}\n"
            f"📝 요약: {meta.get('요약','')}\n"
        )

        if law_raw and law_raw not in ("정보없음", "없음"):
            sources.append({"유형": typ, "관련법률": law_raw})

    return "\n---\n".join(blocks), sources


def scripts_to_text(scripts: Optional[List[Dict[str, str]]], max_lines: int = 60) -> str:
    if not scripts:
        return ""
    lines = []
    for s in scripts[:max_lines]:
        spk = s.get("speaker", "")
        txt = s.get("text", "")
        lines.append(f"{spk}: {txt}")
    return "\n".join(lines)

def keyword_additional_laws(question: str, scripts_text: str, allowed_from_q: set[str]) -> str:
    """키워드 기반 추가 법률 설명(본문에만 쓰는 참고 섹션).
       질문에서 키워드가 잡혔으면 질문만 보고, 아니면 질문+스크립트를 합쳐서 본다."""
    hay = (question or "") if allowed_from_q else ((question or "") + "\n" + (scripts_text or ""))
    out = []

    if any(k in hay for k in ["성희롱", "음란", "음담"]):
        out.append("📚 성희롱 관련 법률:\n- 성폭력범죄의 처벌 등에 관한 특례법 제13조: 2년 이하 징역 또는 2천만원 이하 벌금")
    if any(k in hay for k in ["욕설", "협박","폭언"]):
        out.append("📚 욕설·협박·폭언 관련 법률:\n- 형법 제283조(협박): 3년 이하 징역 또는 500만원 이하 벌금\n- 형법 제260조(폭행): 2년 이하 징역 또는 500만원 이하 벌금")
    if any(k in hay for k in ["모욕", "명예훼손","폭언"]):
        out.append("📚 명예훼손·모욕·폭언 관련 법률:\n- 형법 제307조(명예훼손): 2년 이하 징역 또는 500만원 이하 벌금\n- 형법 제311조(모욕): 1년 이하 징역 또는 200만원 이하 벌금")
    if "업무방해" in hay:
        out.append("📚 업무방해 관련 법률:\n- 형법 제314조(업무방해): 5년 이하 징역 또는 1천5백만원 이하 벌금")
    if "강요" in hay:
        out.append("📚 강요 관련 법률:\n- 형법 제324조(강요): 5년 이하 징역 또는 3천만원 이하 벌금")
    if any(k in hay for k in ["장난전화", "괴롭힘"]):
        out.append("📚 장난전화 관련 법률:\n- 경범죄처벌법 제3조 제1항 제40호: 10만원 이하 벌금, 구류, 과료")
    if "스토킹" in hay:
        out.append("📚 스토킹 관련 법률:\n- 스토킹범죄의 처벌 등에 관한 법률 제18조 제1항: 3년 이하 징역 또는 3천만원 이하 벌금")

    return "\n---\n".join(out)

def is_ask_customer_said(text: str) -> bool:
    return any(k in text for k in ["뭐라고 했", "무슨 말", "고객", "한 말"])

def build_prompts(
    mem_text: str,
    rag_text: str,
    scripts_text: str,
    question: str,
    add_laws_text: str,
    allowed_types: Optional[set[str]] = None
) -> Tuple[str, str]:
    """단일 프롬프트. 스크립트 우선, RAG/추가법률/메모리는 보조."""
    sys = (
        "너는 악성민원 대응/법률 자문 전문가 AI다. 반드시 JSON만 출력한다. "
        "키는 answer, sourcePages 고정. 코드블록/추가설명 금지. "
        "답변 생성 시 반드시 [대화 스크립트]를 우선 근거로 삼고, 사용자의 질문은 이 스크립트에 이어지는 추가 맥락으로만 해석하라."
        "만약 스크립트와 질문이 충돌할 경우 스크립트를 신뢰하라. "
        "참고자료를 answer에 그대로 복붙하지 말고 요약/해설하라. "
        "한국어로만 답하고, 불확실한 내용은 단정하지 말고 '~일 수 있습니다' 같은 완곡 표현을 사용하라. "
        f"아래 허용유형 외의 유형/법률/표현(키워드 포함)은 언급하지 마라. 허용유형: {', '.join(sorted(allowed_types)) if allowed_types else '제한 없음'} "
    )
    user = f"""
아래 자료(스크립트/메모리/RAG/추가법률)를 바탕으로 **JSON으로만** 답변해.

- answer: **정확히 2문단**
  1) 1문단: 즉시 취해야 할 구체적 조치(보고·기록·심리안정·차단/선종료 기준 등)와 실무 팁을 **4~6문장**으로 서술.
  2) 2문단: 아래 문장으로 **반드시 시작** —
     **"당신이 상담한 내용은 ‘{{유형명}}’에 해당할 수 있으며, 관련 법률로는 ‘{{법률명 조문번호}}’가 있습니다."**
     이어서 각 법률을 **한 줄씩** 설명하되, **법률명만 굵게(예: - **형법 제307조**: …)** 표시하고 설명 문구는 굵게 하지 마.
- sourcePages: [{{
    "유형": "<악성민원 유형>",
    "관련법률": "<법률명 제n조>"
  }}] 의 배열만 작성. **마크다운/따옴표/괄호 설명 금지**. (예: "형법 제307조" OK, "형법 제307조(명예훼손)" 금지)

- 참고자료가 부족해도 실제 **유형/법률명을 반드시 채워 넣어라**(합리적 추론). 확실치 않으면 "~일 수 있습니다" 같은 완곡 표현 사용.

[대화 스크립트]
{scripts_text or "(없음)"}

[대화 메모리]
{mem_text}

[참고 법률 자료]
{rag_text}

[키워드 기반 추가 법률]
{add_laws_text or "(없음)"}

[질문]
{question}
"""
    return sys, user


# ---- unified endpoint ----
@router.post("/stream")
async def callchat_stream(body: StreamQuery):
    # (0) 스몰토크면 즉시 종료
    if is_smalltalk(body.question):
        async def smalltalk_events():
            payload = {"answer": smalltalk_reply(body.question), "sourcePages": []}
            yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [END]\n\n"
        return EventSourceResponse(smalltalk_events())

    # (0-1) 특정 질문 → 특정 응답 (캔드 응답)
    canned_tag = match_canned_tag(body.question)
    if canned_tag:
        payload = build_canned_payload(canned_tag)
        async def canned_events():
            async for chunk in slow_emit_json(payload, min_wait=1.8, max_wait=3.0):
                yield chunk
        return EventSourceResponse(canned_events())

    # (0-2) 고객 발화 그대로 묻는 경우 → 스크립트 직접 반환
    if is_ask_customer_said(body.question):
        inbound_lines = [s.get("text","") for s in (body.context_scripts or []) if s.get("speaker")=="INBOUND"]
        answer = "고객 발화 내용은 다음과 같습니다:\n- " + "\n- ".join(inbound_lines) if inbound_lines else "기록된 고객 발화가 없습니다."
        async def direct_events():
            payload = {"answer": answer, "sourcePages": []}
            yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [END]\n\n"
        return EventSourceResponse(direct_events())

    key = ns_key(body.session_id)

    # (1) 메모리
    mem = session_memory[key]
    mem_text = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in mem]) if mem else "(이전 대화 없음)"

    # (2) 스크립트
    scripts_text = scripts_to_text(body.context_scripts)

    # (2-1) 허용 유형 감지: 질문에서 먼저, 없으면 스크립트에서
    allowed_from_q = detect_allowed_types(body.question)
    allowed_from_s = detect_allowed_types(scripts_text)
    allowed_types = allowed_from_q if allowed_from_q else allowed_from_s

    # (3) RAG 검색: 허용유형 필터 적용
    haystack = (scripts_text + "\n" + body.question).strip()
    rag_text, source_pages_rag = retrieve_context(haystack, allowed_types=allowed_types)

    # (4) 본문 보조 설명(선택) — 질문 우선
    add_laws_text = keyword_additional_laws(body.question, scripts_text, allowed_from_q=allowed_from_q)

    # (5) 프롬프트
    sys, user = build_prompts(mem_text, rag_text, scripts_text, body.question, add_laws_text, allowed_types=allowed_types)

    # (6) 모델 스트리밍
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        stream=True
    )

    async def gen():
        full = ""
        try:
            async with session_locks[key]:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full += delta
                        yield f"data: {delta}\n\n"

                # --- 병합 로직 ---
                # 1) 키워드 기반 1차 힌트: 질문 우선
                kw_hay = body.question if allowed_from_q else (scripts_text + "\n" + body.question).strip()
                kw_sources = keyword_pairs_first(kw_hay)

                # 2) 모델 파싱 (JSON-only 아님을 대비해 안전 처리)
                model_answer_raw, model_sources = parse_model_json(full)

                # 3) RAG 소스
                rag_sources = source_pages_rag

                # 4) 최종 병합 (키워드 → RAG → 모델)
                merged_sources = merge_sources(kw_sources, rag_sources, model_sources, limit=3)

                # 4-1) 허용유형 밖 소스 제거
                final_sources = filter_sources_by_types(merged_sources, allowed_types)

                # 5) answer를 2문단 구조로 보정
                final_answer = ensure_two_paragraphs(model_answer_raw or full, final_sources)

                # 5-1) 최종 본문을 허용유형/법률로 Sanitizing + 금지 키워드 치환
                final_answer = sanitize_answer_by_allowed(final_answer, final_sources, allowed_types)

                payload = {
                    "answer": final_answer,
                    "sourcePages": final_sources,
                    "sourcePagesText": format_sourcepages_text(final_sources),
                }
                yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [END]\n\n"

                # (메모리 업데이트)
                mem.append({"q": body.question, "a": final_answer})

        except Exception:
            fail = json.dumps({"answer": "일시적 오류가 발생했습니다.", "sourcePages": []}, ensure_ascii=False)
            yield f"data: [JSON]{fail}\n\n"
            yield "data: [END]\n\n"

    # SSE 버퍼링 방지 헤더
    return EventSourceResponse(gen(), headers={"X-Accel-Buffering": "no"})
