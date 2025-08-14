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
session_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

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

def _ok(v: Optional[str]) -> bool:
    v = (v or "").strip()
    return bool(v) and v not in ("없음", "정보없음")

def legal_like(s: str) -> bool:
    """정책/지침/조례 등은 제외하고, 법률/조문 형태만 허용하기 위한 간단한 필터."""
    low = (s or "").lower()
    exclude_kw = ["정책", "체계", "가이드", "지침", "매뉴얼", "계획", "조례"]
    if any(k in low for k in exclude_kw):
        return False
    return True  # 간단 필터: 법률명/조문 텍스트는 허용

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
            if not legal_like(typ + " " + law):
                continue
            key = (typ, law)
            if key in seen:
                continue
            seen.add(key)
            out.append({"유형": typ, "관련법률": law})
    return out

def retrieve_context(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    """Pinecone RAG: 본문+메타 포함, 중복/저신뢰 제거."""
    emb = client.embeddings.create(input=[query], model="text-embedding-3-small" ).data[0].embedding
    results = index.query(vector=emb, top_k=top_k, include_metadata=True, include_values=False)

    blocks, sources, seen = [], [], set()
    for m in results.get("matches", []):
        if m.get("score", 0) < 0.2:
            continue
        meta = m.get("metadata", {}) or {}
        typ = (meta.get("유형") or "").strip()
        law = (meta.get("관련 법률") or "").strip()
        if not _ok(typ) and not _ok(law):
            continue

        dedup_key = (typ, law)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        blocks.append(
            f"📌 유형: {typ or '정보없음'}\n"
            f"📖 본문: {meta.get('본문','')}\n"
            f"⚖ 관련 법률: {law or '정보없음'}\n"
            f"📝 요약: {meta.get('요약','')}\n"
        )
        # 최종 JSON에서는 '관련법률'(띄어쓰기 없음) 키로 통일
        sources.append({"유형": typ or "정보없음", "관련법률": law or "정보없음"})
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

def law_hints_from_text(text: str) -> List[Dict[str, str]]:
    """룰 기반 최소 보강: 질문/스크립트 키워드에서 대표 조문 힌트."""
    t = (text or "").lower()
    hints: List[Dict[str, str]] = []

    def add(u: str, l: str):
        hints.append({"유형": u, "관련법률": l})

    if any(k in t for k in ["성희롱", "음란", "음담"]):
        add("성희롱/음란발언", "성폭력범죄의 처벌 등에 관한 특례법 제13조")
    if any(k in t for k in ["욕설", "협박"]):
        add("모욕/협박/폭행", "형법 제283조, 제260조")
    if any(k in t for k in ["모욕", "명예훼손"]):
        add("명예훼손/모욕", "형법 제307조, 제311조")
    if "업무방해" in t:
        add("업무방해", "형법 제314조")
    if "강요" in t:
        add("강요", "형법 제324조")
    if any(k in t for k in ["장난전화", "괴롭힘"]):
        add("장난전화/경범", "경범죄처벌법 제3조 제1항 제40호")
    if "스토킹" in t:
        add("스토킹", "스토킹범죄의 처벌 등에 관한 법률 제18조")
    return hints[:5]

def keyword_additional_laws(question: str, scripts_text: str) -> str:
    """키워드 기반 추가 법률 설명(본문에만 쓰는 참고 섹션)."""
    hay = (question or "") + "\n" + (scripts_text or "")
    out = []

    if any(k in hay for k in ["성희롱", "음란", "음담"]):
        out.append("📚 성희롱 관련 법률:\n- 성폭력범죄의 처벌 등에 관한 특례법 제13조: 2년 이하 징역 또는 2천만원 이하 벌금")
    if any(k in hay for k in ["욕설", "협박"]):
        out.append("📚 욕설·협박 관련 법률:\n- 형법 제283조(협박): 3년 이하 징역 또는 500만원 이하 벌금\n- 형법 제260조(폭행): 2년 이하 징역 또는 500만원 이하 벌금")
    if any(k in hay for k in ["모욕", "명예훼손"]):
        out.append("📚 명예훼손·모욕 관련 법률:\n- 형법 제307조(명예훼손): 2년 이하 징역 또는 500만원 이하 벌금\n- 형법 제311조(모욕): 1년 이하 징역 또는 200만원 이하 벌금")
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
        "사실 판단은 [대화 스크립트]를 최우선으로 하고, 충돌 시 스크립트를 신뢰하라. "
        "참고자료를 answer에 그대로 복붙하지 말고 요약/해설하라. "
        "sourcePages에는 '유형/관련법률'만 넣고, 정책·지침·가이드·조례 등은 sourcePages에 담지 말고 answer 본문에서 언급하라."
    )
    user = f"""
아래 자료(스크립트/메모리/RAG/추가법률)를 바탕으로 JSON으로만 답변해.
- answer: 2문단(일반 설명 + "{{유형/관련법률}}" 연결 문장, 필요 시 정책/조례/지침은 본문에 보조로 언급)
- sourcePages: '유형','관련법률' 배열 (법률/조문 중심, 정책/지침/조례는 제외)

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

def clean_source_pages(entries) -> List[Dict[str, str]]:
    """모델이 준 sourcePages 정제 + 정책성 제거."""
    if not isinstance(entries, list):
        return []
    cleaned = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        cleaned.append({
            "유형": (e.get("유형") or "").strip(),
            "관련법률": (e.get("관련법률") or "").strip()
        })
    # 유효/법률형만 필터
    return [e for e in cleaned if _ok(e["유형"]) and _ok(e["관련법률"]) and legal_like(e["유형"] + " " + e["관련법률"])]

# ---- unified endpoint ----
@router.post("/stream")
async def callchat_stream(body: StreamQuery):
    key = ns_key(body.session_id)

    # (1) 메모리
    mem = session_memory[key]
    mem_text = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in mem]) if mem else "(이전 대화 없음)"

    # (2) 스크립트
    scripts_text = scripts_to_text(body.context_scripts)

    # (3) RAG
    rag_text, source_pages_rag = retrieve_context(body.question)

    # (4) 키워드 기반 추가 법률(본문 보조 설명용)
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

                # (7) --- sourcePages 생성 우선순위 ---
                # 1) 키워드 기반(최상위 신뢰)
                kw_sources = law_hints_from_text((scripts_text + "\n" + body.question).strip())

                # 2) 모델 결과(정제)
                gpt_sources_clean = []
                answer_text = full
                try:
                    parsed = json.loads(full)
                    answer_text = parsed.get("answer", full)
                    gpt_sources_clean = clean_source_pages(parsed.get("sourcePages"))
                except Exception:
                    pass

                # 3) RAG(정책/조례 등 제거된 상태: retrieve_context에서 이미 정리)
                rag_sources_clean = clean_source_pages(source_pages_rag)

                # 4) 병합: kw → gpt → rag
                out_source = merge_unique(kw_sources, gpt_sources_clean, rag_sources_clean)

                # 5) 그래도 비면 최소 힌트 유지(kw_sources 그대로)
                if not out_source:
                    out_source = merge_unique(kw_sources)

                payload = {"answer": answer_text, "sourcePages": out_source}
                yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [END]\n\n"

                mem.append({"q": body.question, "a": answer_text})

        except Exception:
            fail = json.dumps({"answer": "일시적 오류가 발생했습니다.", "sourcePages": []}, ensure_ascii=False)
            yield f"data: [JSON]{fail}\n\n"
            yield "data: [END]\n\n"

    return EventSourceResponse(gen())
