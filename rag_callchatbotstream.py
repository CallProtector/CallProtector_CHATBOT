import os, json, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone

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


# ---- smalltalk ----
SMALLTALK_KWS = [
    "안녕","안뇽","하이","hi","hello","헬로","헤이","방가","ㅎㅇ","그냥",
    "잘 지내","뭐해","심심해","심심","ㅎㅎ","ㅋㅋ","굿모닝","굿밤","잘자","좋은 아침","수고","고마워","땡큐","감사","thanks","thx","ㄳ"
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
    return "안녕하세요! 편하게 말씀해 주세요. 민원/상담 관련도 좋고, 일반적인 질문도 환영해요."

# ---- 법률명 정규화/후처리 ----
import re

def _normalize_law_name(law: str) -> str:
    if not law:
        return ""
    # 괄호/주석 제거, 공백 정리
    return re.sub(r"\s*\(.*?\)", "", law).strip()

def _ok(v: str | None) -> bool:
    v = (v or "").strip()
    return bool(v) and v not in ("없음", "정보없음")

def _post_filter_sources(sources: list[dict], limit: int = 3) -> list[dict]:
    """
    - 법률명 기준 dedup (유형 달라도 같은 법률이면 1개만)
    - '없음' 제거, 괄호 제거, ';' ',' 분할
    - 최대 limit개 유지
    """
    out, seen = [], set()
    for e in sources or []:
        typ = (e.get("유형") or "").strip()
        raw = (e.get("관련법률") or "").strip()
        if not (_ok(typ) and _ok(raw)):
            continue
        # 여러 개 한 줄일 수 있음
        for lw in [x.strip() for x in re.split(r"[;,]", raw) if x.strip()]:
            norm = _normalize_law_name(lw)
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"유형": typ, "관련법률": norm})
            if len(out) >= limit:
                return out
    return out

# ---- 키워드 → (유형,법률) 1차 힌트 ----
def keyword_pairs_first(text: str) -> list[dict]:
    hay = (text or "")
    out = []
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
        add("반복(고질.강성민원)", "경범죄처벌법 제3조 제1항 제40호")
    if "스토킹" in hay:
        add("스토킹","스토킹범죄의 처벌 등에 관한 법률 제18조")
    return out[:5]


# ---- 2문단 보정(두 번째 문단 = 유형·법률 연결문 + 법률별 1줄 요약) ----
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
    "국민권익위원회 상담사 보호 지침": "상담 과정에서 발생하는 욕설·폭언·성희롱 등 악·강성 민원으로부터 상담사를 보호하기 위해 마련된 제도적 지침입니다. 상담 종료 기준, 기록 관리, 보호 조치 절차 등을 규정합니다."
}
def _brief_for_law(law: str) -> str:
    # 사전에 없으면 간략 fallback
    if law in _LAW_BRIEFS: return _LAW_BRIEFS[law]
    low = (law or "").lower()
    if "협박" in low: return "협박 행위 전반을 처벌합니다."
    if "폭행" in low: return "타인에 대한 유형력 행사(폭행)를 처벌합니다."
    if "모욕" in low: return "공연한 모욕을 처벌합니다."
    if "명예훼손" in low: return "허위/사실 적시 명예훼손을 처벌합니다."
    if "업무방해" in low: return "업무 수행을 방해하는 행위를 처벌합니다."
    if "스토킹" in low: return "지속·반복적 스토킹을 처벌합니다."
    return "관련 행위를 규율·제재하여 피해 방지를 도모합니다."

def _build_second_paragraph(sources: list[dict]) -> str:
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
    # ✅ 법률명만 굵게
    bullets = "\n".join([f"- **{l}**: {_brief_for_law(l)}" for l in laws])
    return head + "\n" + bullets

def ensure_two_paragraphs(answer: str, sources: list[dict]) -> str:
    text = (answer or "").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = ["상황 기록, 증거 보존, 상급자 보고, 심리 안정 확보 등 즉시 조치를 진행하세요."]
    second = _build_second_paragraph(sources)
    if len(paras) == 1:
        paras.append(second)
    else:
        paras[1] = second
    # 1문단이 너무 짧으면 보강
    first = paras[0]
    sents = [s for s in re.split(r"[.。]\s*", first) if s.strip()]
    if len(sents) < 4:
        supplement = (" 통화 선종료·차단 기준을 숙지하고, 재발 방지를 위한 안내 멘트를 사용하세요. "
                      "내부 시스템에 시간/상황/발언을 구체 기록하고 즉시 보호조치를 요청하세요.")
        paras[0] = (first + supplement).strip()
    return "\n\n".join(paras)


def merge_unique(*lists: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """(유형, 관련법률) 기준으로 앞쪽 리스트 우선 병합."""
    seen = set()
    out: List[Dict[str, str]] = []
    for lst in lists:
        for e in lst or []:
            if not isinstance(e, dict):
                continue
            typ = (e.get("유형") or "").strip()
            law = (e.get("관련법률") or "").strip()
            if not (_ok(typ) and _ok(law)):
                continue
            key = (typ, law)
            if key in seen:
                continue
            seen.add(key)
            out.append({"유형": typ, "관련법률": law})
    return out


def _format_sourcepages_pairs(sources: list[dict]) -> str:
    """유형–관련법률 쌍을 블록으로 묶어, 블록 사이에 빈 줄을 넣어 반환."""
    blocks = []
    for e in sources or []:
        t = (e.get("유형") or "").strip()
        l = (e.get("관련법률") or "").strip()
        if not t or not l:
            continue
        blocks.append(f"- 유형: {t}\n- 관련법률: {l}")
    return "\n\n".join(blocks)

def retrieve_context(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    emb = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    results = index.query(vector=emb, top_k=top_k, include_metadata=True, include_values=False)

    blocks, sources, seen = [], [], set()
    matches = getattr(results, "matches", None) or (results.get("matches", []) if isinstance(results, dict) else [])

    for m in matches:
        score = getattr(m, "score", None)
        if score is None and isinstance(m, dict):
            score = m.get("score", 0)
        if (score or 0) < 0.2:
            continue

        meta = getattr(m, "metadata", None) or (m.get("metadata", {}) if isinstance(m, dict) else {}) or {}
        typ = (meta.get("유형") or "").strip()
        law_raw = (meta.get("관련 법률") or "").strip()

        # 화면용 블록
        blocks.append(
            f"📌 **유형**: {typ or '정보없음'}\n"
            f"📖 본문: {meta.get('본문','')}\n"
            f"⚖ **관련 법률**: {law_raw or '정보없음'}\n"
            f"📝 요약: {meta.get('요약','')}\n"
        )
        # JSON용 sourcePages (정규화 + 분할 + dedup)
        if _ok(law_raw):
            for lw in [x.strip() for x in re.split(r"[;,]", law_raw) if x.strip()]:
                norm = _normalize_law_name(lw)
                key = (typ, norm)
                if not _ok(typ) or not _ok(norm) or key in seen:
                    continue
                seen.add(key)
                sources.append({"유형": typ, "관련법률": norm})
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

def keyword_additional_laws(question: str, scripts_text: str) -> str:
    """키워드 기반 추가 법률 설명(본문에만 쓰는 참고 섹션)."""
    hay = (question or "") + "\n" + (scripts_text or "")
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

def build_prompts(mem_text: str, rag_text: str, scripts_text: str, question: str, add_laws_text: str) -> Tuple[str, str]:
    """단일 프롬프트. 스크립트 우선, RAG/추가법률/메모리는 보조."""
    sys = (
        "너는 악성민원 대응/법률 자문 전문가 AI다. 반드시 JSON만 출력한다. "
        "키는 answer, sourcePages 고정. 코드블록/추가설명 금지. "
        "답변 생성 시 반드시 [대화 스크립트]를 우선 근거로 삼고, 사용자의 질문은 이 스크립트에 이어지는 추가 맥락으로만 해석하라."
        "만약 스크립트와 질문이 충돌할 경우 스크립트를 신뢰하라. "
        "참고자료를 answer에 그대로 복붙하지 말고 요약/해설하라. "
        "한국어로만 답하고, 불확실한 내용은 단정하지 말고 '~일 수 있습니다' 같은 완곡 표현을 사용하라."
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

    key = ns_key(body.session_id)

    # (1) 메모리
    mem = session_memory[key]
    mem_text = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in mem]) if mem else "(이전 대화 없음)"

    # (2) 스크립트
    scripts_text = scripts_to_text(body.context_scripts)

    # (3) RAG
    rag_text, source_pages_rag = retrieve_context(body.question)

    # (4) 본문 보조 설명(선택): 기존 keyword_additional_laws 유지
    add_laws_text = keyword_additional_laws(body.question, scripts_text)

    # (5) 프롬프트
    sys, user = build_prompts(mem_text, rag_text, scripts_text, body.question, add_laws_text)

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
                # 1) 키워드 기반 1차 힌트
                kw_sources = keyword_pairs_first((scripts_text + "\n" + body.question).strip())

                # 2) 모델 파싱
                model_answer = full
                model_sources = []
                try:
                    parsed = json.loads(full)
                    if isinstance(parsed, dict):
                        model_answer = parsed.get("answer", model_answer)
                        if isinstance(parsed.get("sourcePages"), list):
                            model_sources = [
                                {"유형": (e.get("유형") or "").strip(),
                                 "관련법률": _normalize_law_name((e.get("관련법률") or "").strip())}
                                for e in parsed["sourcePages"]
                                if isinstance(e, dict)
                            ]
                except Exception:
                    pass

                # 3) RAG 소스는 이미 정규화됨
                rag_sources = source_pages_rag

                # 4) 우선순위 병합: 키워드 → 모델 → RAG
                merged = merge_unique(kw_sources, model_sources, rag_sources)

                # 5) 최종 후처리(분할/정규화/중복 제거/최대 3개)
                final_sources = _post_filter_sources(merged, limit=3)

                # 6) answer를 2문단 구조로 보정
                final_answer = ensure_two_paragraphs(model_answer, final_sources)

                payload = {"answer": final_answer, "sourcePages": final_sources, "sourcePagesText": _format_sourcepages_pairs(final_sources)}
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

