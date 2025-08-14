import os
import json
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone

# ✅ 1. 환경 변수 로드
load_dotenv()

# ✅ 2. 라우터 초기화
router = APIRouter()

# ✅ 3. OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 오레곤(us-west-2) 인덱스 사용
index_name = os.getenv("PINECONE_INDEX", "legal-guideline-usw2")
index = pc.Index(index_name)

# ✅ 4. 요청 모델 정의
class Query(BaseModel):
    session_id: int
    question: str

# ---------- 유틸: 키워드 기반 법률쌍(1차 우선) ----------
def keyword_pairs_first(text: str):
    """
    질문(또는 스크립트)에서 키워드를 감지해
    sourcePages에 먼저 들어갈 {유형,관련법률} 쌍을 리턴.
    """
    hay = (text or "")
    out = []

    def add(u, l):
        out.append({"유형": u, "관련법률": l})

    if any(k in hay for k in ["성희롱", "음란", "음담"]):
        add("성희롱/음란발언", "성폭력범죄의 처벌 등에 관한 특례법 제13조")
    if any(k in hay for k in ["욕설", "협박"]):
        add("협박/폭행 가능성", "형법 제283조(협박); 형법 제260조(폭행)")
    if any(k in hay for k in ["모욕", "명예훼손"]):
        add("명예훼손·모욕", "형법 제307조(명예훼손); 형법 제311조(모욕)")
    if "업무방해" in hay:
        add("업무방해", "형법 제314조")
    if "강요" in hay:
        add("강요", "형법 제324조")
    if any(k in hay for k in ["장난전화", "괴롭힘"]):
        add("장난전화/경범", "경범죄처벌법 제3조 제1항 제40호")
    if "스토킹" in hay:
        add("스토킹", "스토킹범죄의 처벌 등에 관한 법률 제18조 제1항")

    # 너무 길어지지 않게 상위 3개만
    return out[:3]

def _clean_pair(e):
    if not isinstance(e, dict):
        return None
    t = (e.get("유형") or "").strip()
    l = (e.get("관련법률") or "").strip()
    if not t or not l:
        return None
    return {"유형": t, "관련법률": l}

def _merge_sources(primary, *others):
    """
    primary → others 순으로 합치며 (유형,관련법률) 중복 제거.
    """
    seen = set()
    merged = []

    def push_list(lst):
        for e in lst or []:
            ce = _clean_pair(e)
            if not ce:
                continue
            key = (ce["유형"], ce["관련법률"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(ce)

    push_list(primary)
    for o in others:
        push_list(o)
    return merged

# ✅ 5. 유사 문단 검색 (본문+메타데이터 포함)
def retrieve_context(query: str, top_k: int = 2):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"  # ✅ 더 빠르고 저렴
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    context_blocks = []
    source_pages = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {}) or {}
        typ = (meta.get("유형") or "").strip() or "없음"
        law = (meta.get("관련 법률") or "").strip() or "없음"

        context_blocks.append(
            f"📌 유형: {typ}\n"
            f"📖 본문: {meta.get('본문', '')}\n"
            f"⚖ 관련 법률: {law}\n"
            f"📝 요약: {meta.get('요약', '')}\n"
        )
        # 최종 JSON에서는 '관련법률'(띄어쓰기 없음)
        source_pages.append({"유형": typ, "관련법률": law})

    return "\n---\n".join(context_blocks), source_pages

# ✅ 6. GPT 스트리밍 + JSON 응답 (키워드 기반 법률을 sourcePages 1차 반영)
@router.post("/stream")
async def stream_chat(query: Query):
    # RAG
    context, source_pages_rag = retrieve_context(query.question)

    # 1차: 키워드 기반 법률쌍
    source_pages_keywords = keyword_pairs_first(query.question)

    # 프롬프트
    prompt = f"""
너는 악성민원 대응 및 관련 법률 상담을 도와주는 AI야.
아래 참고 자료를 바탕으로 사용자의 질문에 대해 자연스럽고 자세한 문장으로 답변해줘.

반드시 JSON으로만 출력하고, 코드 블록은 쓰지 마. 모든 출력은 자연스러운 한글이어야 해.

- answer: 두 문단으로 작성
  1) 사용자의 질문에 대한 일반적이고 자연스러운 답변
  2) "당신이 상담한 내용은 ~유형에 포함되며, 관련 법률로는 ~가 있습니다." 형식으로 설명
     (정책·지침·조례 등은 answer 본문에서 보조로 언급해도 되지만, sourcePages에는 법률·조문 위주로 작성)

- sourcePages: 아래 참고자료의 '유형'과 '관련법률'만 배열로 정리

예시:
{{
  "answer": "…",
  "sourcePages": [{{"유형":"반복 민원","관련법률":"국민권익위원회 상담사 보호 지침"}}]
}}

### 참고 자료:
{context}

### 질문:
{query.question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "너는 악성민원 대응 가이드 및 관련 법률 문서를 기반으로 상담하는 전문가 AI다. 반드시 JSON 형식으로만 출력하고, 코드 블록이나 부가 설명은 절대 하지 마."
            },
            {"role": "user", "content": prompt}
        ],
        stream=True
    )

    async def event_generator():
        full_response = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                yield f"data: {delta}\n\n"

        # ----- 모델 출력 JSON 보정 및 sourcePages 우선 병합 -----
        model_answer = full_response
        model_sources = []
        try:
            parsed = json.loads(full_response)
            if isinstance(parsed, dict):
                if "answer" in parsed and isinstance(parsed["answer"], str):
                    model_answer = parsed["answer"]
                sp = parsed.get("sourcePages")
                if isinstance(sp, list):
                    model_sources = [_clean_pair(e) for e in sp if _clean_pair(e)]
        except Exception:
            pass

        # 병합 규칙: 키워드(1차) → 모델 sourcePages → RAG sourcePages
        final_sources = _merge_sources(source_pages_keywords, model_sources, source_pages_rag)

        payload = {"answer": model_answer, "sourcePages": final_sources}
        yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [END]\n\n"

    return EventSourceResponse(event_generator())
