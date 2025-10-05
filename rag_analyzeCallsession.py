# rag_analyzeCallsession.py
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone
import os
import json
import asyncio
from dotenv import load_dotenv
import re

# 공통 정책 모듈 (reply_policy.py)
from reply_policy import (
    keyword_pairs_first,
    parse_model_json,
    merge_sources,            # (키워드 → RAG → 모델) 병합 + post_filter_sources 포함
    ensure_two_paragraphs,    # 2문단 보정
    format_sourcepages_text,  # 화면용 sourcePagesText
)

# 환경 변수 로드
load_dotenv()

router = APIRouter()

# 환경변수 기반 모델명 (하드코딩 방지)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# 오레곤(us-west-2) 인덱스 사용
index_name = os.getenv("PINECONE_INDEX", "legal-guideline-usw2")
index = pc.Index(index_name)

# =========================================
# RAG: 컨텍스트 + sourcePages(원자료) 수집
# =========================================
def retrieve_context(query: str, top_k: int = 5):
    emb = client.embeddings.create(
        input=[query],
        model=EMBED_MODEL
    ).data[0].embedding

    # include_values=False → payload 감소/속도 향상
    results = index.query(vector=emb, top_k=top_k, include_metadata=True, include_values=False)

    context_blocks = []
    source_pages = []

    matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", []) or []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}

        typ = (meta.get("유형") or "없음").strip()
        # ↙️ 메타키가 '관련 법률' 또는 '관련법률' 어느 쪽이든 커버
        law_raw = (meta.get("관련 법률") or meta.get("관련법률") or "없음").strip()

        context_blocks.append(
            f"📌 유형: {typ or '없음'}\n"
            f"📖 본문: {meta.get('본문','')}\n"
            f"⚖ 관련 법률: {law_raw or '없음'}\n"
            f"📝 요약: {meta.get('요약','')}\n"
        )

        if law_raw and law_raw != "없음":
            # ⛳ 원문 그대로 적재(정규화/분할/중복제거는 merge_sources(post_filter_sources)에서 공통 처리)
            source_pages.append({"유형": typ, "관련법률": law_raw})

    return "\n---\n".join(context_blocks), source_pages

# =========================================
# 자유서술 답변에서 법률명 패턴 보완 추출
#  - 괄호 안 조문명까지 허용 (예: 형법 제311조(모욕))
# =========================================
LAW_REGEX = re.compile(
    r"(?:[가-힣A-Za-z·\s]{1,25}?법(?:률)?\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*\([^)]+\))?)"
)

def extract_law_mentions(text: str, limit: int = 5) -> list[str]:
    """모델의 자유서술 답변에서 '○○법 제n조(제m항)' 패턴을 추출."""
    if not text:
        return []
    hits = LAW_REGEX.findall(text)
    out, seen = [], set()
    for raw in hits:
        lw = re.sub(r"\s+", " ", raw).strip()
        key = lw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(lw)
        if len(out) >= limit:
            break
    return out

# =========================================
# 의미 기반 악성 발언 유형 탐지 (정규식/사전)
#  - '모욕' 용어 대신 UI/응답 문구는 '폭언'으로 표기
# =========================================
THREAT_PATTERNS = [
    r"죽(여|인다|여버리|여버린다)", r"가만(두지|안[둘둔다])", r"찾아가(서)? (가만두지|혼내|죽이)",
    r"찌[른를]다", r"폭탄", r"테러", r"협박(한다)?"
]
SEXUAL_PATTERNS = [
    r"야하", r"야하게", r"목소리.*야하", r"섹시", r"외모.*(평가|품평)", r"밤에.*(피는|만나)",
    r"(음란|음탕|음흉)", r"(신음|야동|가슴|엉덩이|음부|유두)"
]
INSULT_PATTERNS = [  # 서비스 정책에 맞게 확장/조정
    r"시발", r"씨발", r"개새끼", r"미친", r"병신", r"등신", r"x발", r"꺼져", r"애미|애비"
]

def detect_abuse_types(text: str) -> set[str]:
    """
    의미 기반 탐지 결과를 {'협박','성희롱','폭언'} 셋으로 반환.
    - 법률 매핑 시 '폭언'은 주로 형법 제314조(업무방해) 및 제307조(명예훼손) 논의와 연결.
    """
    t = text.lower()
    flags = set()
    if any(re.search(p, t) for p in THREAT_PATTERNS):
        flags.add("협박")
    if any(re.search(p, t) for p in SEXUAL_PATTERNS):
        flags.add("성희롱")
    if any(re.search(p, t) for p in INSULT_PATTERNS):
        flags.add("폭언")  # ← '모욕' 대신 '폭언'으로 표기
    return flags

# ---------- 분석 엔드포인트 ----------
@router.post("/analyze")
async def analyze_call_session(request: Request):
    body = await request.json()
    scripts = body.get("scripts", [])

    if not scripts:
        # 간단 에러 SSE
        async def err_stream():
            yield "data: 세션에 스크립트 없음\n\n"
            yield "data: [END]\n\n"
        return EventSourceResponse(err_stream(), status_code=400, headers={"X-Accel-Buffering": "no"})

    # 통화 내용 추출
    context_dialogue = "\n".join(f"{s['speaker']}: {s['text']}" for s in scripts)

    # RAG 질의
    question = (
        "다음 상담 내용에서 고객의 발언 중 위법 소지가 있는 부분이 있다면 "
        "어떤 법률 조항(법률명 + 조문번호 포함)이 적용될 수 있으며, "
        "어떻게 대응해야 하는지 알려줘.\n\n"
        f"{context_dialogue}"
    )

    # RAG 검색
    rag_context, source_pages_rag = retrieve_context(context_dialogue)

    # 의미 기반 유형 탐지 (신규)
    detected = detect_abuse_types(context_dialogue)

    # 추가 법률 정보 (탐지 결과 기반)
    additional_laws = ""
    if "성희롱" in detected:
        additional_laws += (
            "\n📚 성희롱 관련 법률:"
            "\n- 성폭력범죄의 처벌 등에 관한 특례법 제13조(통신매체를 이용한 음란행위): 2년 이하 징역 또는 2천만원 이하 벌금"
        )
    if "협박" in detected:
        additional_laws += (
            "\n📚 협박 관련 법률:"
            "\n- 형법 제283조(협박): 3년 이하 징역 또는 500만원 이하 벌금"
        )
    if "폭언" in detected:
        additional_laws += (
            "\n📚 폭언 관련 법률:"
            "\n- 형법 제311조(모욕): 1년 이하 징역 또는 200만원 이하 벌금"
            "\n- 형법 제307조(명예훼손): 2년 이하 징역 또는 500만원 이하 벌금"
        )

    if additional_laws:
        rag_context += "\n---\n" + additional_laws

    # 프롬프트 (자유서술형 — '모욕' 용어 대신 '폭언' 중심 표기)
    #    예시 블록은 입력에 따라 동적으로 편향을 줄여도 되지만,
    #    여기서는 간결성을 위해 고정 예시 + 폭언 표기로 유지
    prompt = f"""
너는 악성민원 대응 및 관련 법률 자문을 돕는 전문가 AI야.

아래 통화 내용을 참고해서 다음 형식에 맞춰 정중하고 구조화된 요약을 생성해줘.

✅ 특히 **적용 가능한 법률**에는 반드시 '법률명 + 조문번호 + 조문명'을 포함해서 작성하고,
   각 법률이 어떤 악성 발언 유형(예: 성희롱, 폭언, 명예훼손, 협박 등)에 대응되는지 간단히 설명해줘.

✅ 또한 Markdown 문법과 아이콘을 포함하고, 출력은 자연스럽고 띄어쓰기가 올바른 한국어 문장으로 작성해줘.

[응답 형식 예시]
안녕하세요 상담원님, 방금 상담 중 고객으로부터 폭언이나 성희롱 발언을 받으셨네요. 관련 법률과 대응 방법을 안내해드릴게요.

👩‍⚖️ **적용 가능한 법률:**
- **형법 제311조(모욕)**: 상대방을 공개적으로 모욕하거나 심한 욕설을 하는 폭언 상황에 주로 적용됩니다.
- **성폭력범죄의 처벌 등에 관한 특례법 제13조(통신매체를 이용한 음란행위)**: 성적 수치심을 유발하는 발언이 있을 경우 적용될 수 있습니다.
- **형법 제283조(협박)**: 신체·생명 등에 위해를 가하겠다는 취지의 위협 발언에 적용됩니다.

⚖️ **대응 방법:**
📝 1. **사내 대응 절차**
   -  **1차 조치**: 고객 발언이 폭언·성희롱·협박에 해당될 경우 즉시 ARS 경고멘트를 송출하거나 통화 종료 권한을 행사할 수 있습니다.
   -  **2차 조치**: 통화 종료 후, 소속 부서장에게 상황을 **보고**하고, 상담원 보호를 위한 **사내 대응 매뉴얼에 따라 악성민원 등록**을 요청하세요.
   -  **3차 조치**: 필요시 악성민원 전담관리자가 해당 고객의 **재통화 차단**, **주의 고객 등록**, **전담 응대 팀 이관** 등을 검토할 수 있습니다.
   -  **상담 지원**: 정신적 충격이 있는 경우, **EAP 프로그램(심리상담/내부상담센터)** 등을 통해 보호 조치를 받을 수 있습니다.

⚒️ 2. **법적 조치**
   -  **내용 기록**: 폭언, 성희롱, 협박 등이 있었다면 해당 발언 내용을 녹취 및 대화 로그로 보관하고, **상세 보고서 작성**을 권장합니다.
   -  **사내 법무팀/감사팀 협조 요청**: 반복적이거나 악의적인 사례는 법무팀과 협의해 **경고장 발송**, **법률 자문**, **형사고발 여부 검토** 등을 진행할 수 있습니다.
   -  **형사고소 및 민사청구**: 실제 피해 발생 시에는 모욕(폭언)·명예훼손·협박 등으로 고소가 가능하며, 정신적 피해에 따른 **위자료 청구**도 고려할 수 있습니다.

➕ 추가로 도움이 필요하시면 언제든 말씀해주세요!

[❗단, 문제가 되지 않는 평범한 통화라면 "법적 조치 대상은 아니며 일반 민원 응대로 판단됩니다."로 간단히 응답해줘.]

---

# 통화 내용:
{context_dialogue}

# 참고 법률 자료:
{rag_context}
"""

    # GPT 스트리밍 응답
    async def event_generator():
        full_response = ""
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    # SSE 규격상 data: 접두사 권장
                    yield f"data: {delta}\n\n"

            # -----------------------------
            # 병합 파이프라인 (프롬프트 자유서술 유지)
            # -----------------------------
            # 1) 키워드 기반 1차 힌트
            kw_sources = keyword_pairs_first(context_dialogue + "\n" + question)

            # (추가) 의미 기반 탐지 결과를 kw_sources에 주입
            #  - merge_sources에서 관련법률이 비어 있어도 RAG/정규식으로 보강 가능
            for t in detected:  # {'협박','성희롱','폭언'}
                kw_sources.append({"유형": t, "관련법률": ""})

            # 2) 모델 출력 파싱
            #   - JSON이면 JSON 사용
            #   - JSON 아니면 자유 텍스트로 보고, 정규식으로 법률명 추출(보완)
            model_answer_raw, model_sources = parse_model_json(full_response)
            if not model_sources:
                law_hits = extract_law_mentions(full_response)
                if law_hits:
                    inferred_type = (list(detected)[0] if detected else (kw_sources[0]["유형"] if kw_sources else "관련법률"))
                    model_sources = [{"유형": inferred_type, "관련법률": lw} for lw in law_hits]

            # 3) RAG 소스
            rag_sources = source_pages_rag

            # 3.5) 평범한 통화(모든 근거가 비었을 때) 빠른 종료
            if not kw_sources and not model_sources and not rag_sources:
                final_answer = (
                    "안녕하세요 상담원님, 방금 통화 중에 발생한 상황에 대해 처리 방법과 관련 법률을 안내해드리겠습니다.\n\n"
                    "현재 통화 내용에서는 특별히 문제가 되는 발언이 발견되지 않았습니다. "
                    "따라서 본 건은 법적 조치 대상은 아니며 일반 민원 응대로 판단됩니다.\n\n"
                    "➕ 추가로 도움이 필요하시면 언제든 말씀해주세요!"
                )
                payload = {"answer": final_answer, "sourcePages": []}
                yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [END]\n\n"
                return

            # 4) 최종 병합 (키워드 → RAG → 모델) + 정규화/분할/중복제거/limit 처리
            final_sources = merge_sources(kw_sources, rag_sources, model_sources, limit=3)

            # 5) answer 2문단 보정 (모델 JSON answer 우선, 없으면 자유 텍스트)
            final_answer = ensure_two_paragraphs(model_answer_raw or full_response, final_sources)

            payload = {
                "answer": final_answer,
                "sourcePages": final_sources,
                "sourcePagesText": format_sourcepages_text(final_sources),
            }
            yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [END]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [END]\n\n"

    # 버퍼링 방지 헤더 통일
    return EventSourceResponse(event_generator(), headers={"X-Accel-Buffering": "no"})
