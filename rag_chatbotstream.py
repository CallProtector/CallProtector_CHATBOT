import os
import json
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone
import re

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

# ✅ 일상적인 대화(smalltalk)로 분류할 키워드
SMALLTALK_KWS = [
    "안녕", "안뇽", "하이", "hi", "hello", "헬로", "헤이", "방가","ㅎㅇ", "그냥",
    "잘 지내", "뭐해", "심심해","심심","ㅎㅎ", "ㅋㅋ", "굿모닝", "굿밤", "잘자","좋은 아침", "수고", "고마워","땡큐", "감사", "thanks", "thx","ㄳ"
]

# 입력 문장이 smalltalk(일상 대화)인지 판별
def is_smalltalk(text: str) -> bool:
    t = (text or "").strip().lower()
    # 키워드 중 하나라도 들어가면 무조건 smalltalk
    return any(k in t for k in SMALLTALK_KWS)

# smalltalk 유형에 따라 적절한 답변 생성
def smalltalk_reply(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["안녕", "안뇽", "하이", "hello", "hi", "헬로", "헤이","방가","ㅎㅇ"]):
        return "안녕하세요! 만나서 반가워요 😊 무엇을 도와드릴까요?"
    if any(k in t for k in ["굿모닝","좋은 아침"]):
        return "안녕하세요! 잘 지내셨나요? 😊 무엇을 도와드릴까요?"
    if any(k in t for k in ["굿밤","잘자"]):
        return "고마워요! 편안한 밤 되세요 🌛"
    if any(k in t for k in ["고마워", "감사", "땡큐","thx", "thanks","수고","ㄳ"]):
        return "별말씀을요! 도움이 되어 기뻐요. 또 궁금한 점 있으면 편하게 물어보세요."
    if any(k in t for k in ["뭐해","심심해","심심"]):
        return "여기 있어요! 질문을 기다리는 중이에요. 어떤 도움이 필요하신가요?"
    if any(k in t for k in ["ㅎㅎ", "ㅋㅋ","그냥"]):
        return "헤헤 😄 농담도 좋아요. 이제 본론으로—무엇을 도와드릴까요?"
    # 기본
    return "안녕하세요! 편하게 말씀해 주세요. 민원/상담 관련도 좋고, 일반적인 질문도 환영해요."

# ✅ 4. 요청 모델 정의  (클라이언트에서 들어오는 데이터 형식)
class Query(BaseModel):
    session_id: int
    question: str

# ---------- 유틸: 키워드 기반 법률쌍(1차 우선) ----------

# 질문에서 키워드 감지 후 (유형, 관련법률) 쌍 추출 (1차 매핑)
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
    if any(k in hay for k in ["욕설", "협박", "폭언"]):
        add("협박/폭행(폭언) 가능성", "형법 제283조(협박); 형법 제260조(폭행)")
    if any(k in hay for k in ["모욕", "명예훼손", "폭언"]):
        add("명예훼손·모욕·폭언", "형법 제307조(명예훼손); 형법 제311조(모욕)")
    if "업무방해" in hay:
        add("업무방해", "형법 제314조")
    if "강요" in hay:
        add("강요", "형법 제324조")
    if any(k in hay for k in ["장난전화", "괴롭힘"]):
        add("장난전화/경범", "경범죄처벌법 제3조 제1항 제40호")
    if any(k in hay for k in ["반복적인 민원"]):
        add("반복(고질.강성민원)", "경범죄처벌법 제3조 제1항 제40호")
    if "스토킹" in hay:
        add("스토킹", "스토킹범죄의 처벌 등에 관한 법률 제18조 제1항")

    # 너무 길어지지 않게 상위 3개만
    return out[:3]

# sourcePages 항목 정리 (빈 값 제거, 공백 제거)
def _clean_pair(e):
    if not isinstance(e, dict):
        return None
    t = (e.get("유형") or "").strip()
    l = (e.get("관련법률") or "").strip()
    if not t or not l:
        return None
    return {"유형": t, "관련법률": l}

 # 여러 sourcePages 리스트를 합치고 중복 제거
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

# ✅ 법률 한 줄 요약 사전 (특정 조항 설명)
_LAW_BRIEFS = {
    "성폭력범죄의 처벌 등에 관한 특례법 제13조": "통신수단을 이용한 음란·성적 수치심 유발 행위를 처벌합니다. 이는 2년 이하 징역 또는 2천만원 이하 벌금형에 해당합니다. ",
    "형법 제283조": "폭행·협박으로 상대방의 의사결정을 제압하는 행위를 처벌합니다. 이는  3년 이하 징역 또는 500만원 이하 벌금형에 해당합니다.",
    "형법 제260조": "상대방의 신체에 대해 유형력을 행사하는 폭행을 처벌합니다. 이는 2년 이하 징역 또는 500만원 이하 벌금형에 해당합니다.",
    "형법 제307조": "허위 사실 적시 또는 사실 적시로 타인의 명예를 훼손하는 행위를 처벌합니다. 이는 2년 이하 징역 또는 500만원 이하 벌금형에 해당합니다.",
    "형법 제311조": "공연히 사람을 모욕하는 행위를 처벌합니다. 이는 1년 이하 징역 또는 200만원 이하 벌금형에 해당합니다.",
    "형법 제314조": "위력 또는 기타 방법으로 타인의 업무를 방해하는 행위를 처벌합니다. 이는 5년 이하 징역 또는 1천5백만원 이하 벌금형에 해당합니다.",
    "형법 제324조": "폭행·협박 등으로 의사에 반해 의무 없는 일을 하게 하는 강요를 처벌합니다. 이는 5년 이하 징역 또는 3천만원 이하 벌금형에 해당합니다.",
    "경범죄처벌법 제3조 제1항 제40호": "정당한 이유 없이 반복적 전화 등으로 남을 괴롭히는 행위를 제재합니다. 이는 10만원 이하 벌금, 구류, 과료형에 해당합니다.",
    "스토킹범죄의 처벌 등에 관한 법률 제18조 제1항": "지속적·반복적 스토킹 범죄를 처벌하고 보호조치를 규정합니다. 이는 3년 이하 징역 또는 3천만원 이하 벌금형에 해당합니다.",
    "국민권익위원회 상담사 보호 지침": "상담 과정에서 발생하는 욕설·폭언·성희롱 등 악·강성 민원으로부터 상담사를 보호하기 위해 마련된 제도적 지침입니다. 상담 종료 기준, 기록 관리, 보호 조치 절차 등을 규정합니다.",
    "민원처리법 제23조": "동일·반복 민원에 대한 처리 제한/종결 절차를 규정합니다. 기관 지침에 따라 반복 제기에 대해 종결할 수 있습니다."
}

# 키워드 기반 기본 요약(매핑 없을 때 중복 최소화)
# 사전에 없는 법률명을 키워드 기반으로 간단 설명 생성
def _brief_fallback_by_keyword(law: str) -> str:
    if "협박" in law:
        return "상대방에게 공포심을 야기하는 협박 행위를 처벌합니다."
    if "폭행" in law:
        return "상대방 신체에 대한 유형력 행사(폭행)를 처벌합니다."
    if "모욕" in law:
        return "공연히 사람을 모욕하는 언행을 처벌합니다."
    if "명예훼손" in law:
        return "허위 사실 또는 사실 적시의 명예훼손 행위를 처벌합니다."
    if "통신" in law or "이용음란" in law or "성폭력" in law:
        return "통신수단을 이용한 성적 수치심 유발 행위를 처벌합니다."
    if "업무방해" in law:
        return "위력 기타 방법으로 타인의 업무를 방해하는 행위를 처벌합니다."
    if "스토킹" in law:
        return "지속·반복적 스토킹 행위를 처벌하고 피해자 보호를 규정합니다."
    if "국민권익위원회 상담사 보호 지침" in law:
        return "상담 과정에서 발생하는 욕설·폭언·성희롱 등 악·강성 민원으로부터 상담사를 보호하기 위해 마련된 제도적 지침입니다. 상담 종료 기준, 기록 관리, 보호 조치 절차 등을 규정합니다."
    return "해당 조항은 관련 행위를 규율·제재하여 피해 방지를 도모합니다."


# 법률 요약 설명 반환 (사전 매핑 우선, 없으면 fallback)
def _brief_for_law(law: str) -> str:
    return _LAW_BRIEFS.get(law, _brief_fallback_by_keyword(law))

 # answer의 두 번째 문단을 생성 (유형/법률 나열 + 각 법률 설명)
def _build_second_paragraph(sources: list[dict]) -> str:
    if not sources:
        head = "당신이 상담한 내용은 **‘해당 유형’**에 해당할 수 있으며, 관련 법률로는 **‘관련 법률’**이 있습니다."
        tail = "각 법률의 구체 적용은 상황에 따라 달라질 수 있으므로, 기관 지침과 법률 자문을 함께 참고하시길 권장드립니다."
        return f"{head}\n{tail}"

    typ = (sources[0].get("유형") or "해당 유형").strip()
    # 법률만 모아 중복 제거(순서 유지)
    laws = [e.get("관련법률", "").strip() for e in sources if e and e.get("관련법률")]
    seen, unique_laws = set(), []
    for lw in laws:
        if lw and lw not in seen:
            seen.add(lw)
            unique_laws.append(lw)

    laws_str = "’, ‘".join(unique_laws) if unique_laws else "관련 법률"

    # 머리 문장: 유형/법률 목록만 굵게
    head = f"당신이 상담한 내용은 **‘{typ}’**에 해당할 수 있으며, 관련 법률로는 **‘{laws_str}’**가 있습니다."

    # ✅ 각 항목: **법률명**만 굵게 + 한 줄 설명, 항목 사이 ‘한 줄’ 간격
    lines = [f"- **{law}**: {_brief_for_law(law)}" for law in unique_laws]
    tail = "\n".join(lines) if lines else "상세 적용은 사안의 맥락에 따라 달라질 수 있습니다."

    return f"{head}\n{tail}"


# 답변을 항상 2문단 구조로 보정 (첫 문단 보강, 두 번째 문단 재작성)
def _ensure_two_paragraphs(answer: str, final_sources: list[dict]) -> str:
    text = (answer or "").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paras:
        paras = ["상황 기록, 증거 보존, 상급자 보고, 심리적 안정 확보 등 즉시 조치를 진행하세요."]

    second = _build_second_paragraph(final_sources)

    if len(paras) == 1:
        paras.append(second)
    else:
        # ✅ 무조건 dedup 로직을 거친 결과로 교체
        paras[1] = second

    first_sentences = [s for s in paras[0].split("。") if s.strip()] if "。" in paras[0] else [s for s in paras[0].split(".") if s.strip()]
    if len(first_sentences) < 4:
        supplement = "사건 직후에는 통화 선종료 기준과 차단 방침을 숙지하고, 재발 방지를 위해 안내 멘트를 활용하세요. 내부 기록 시스템에 시간·상황·발언 내용을 구체적으로 남기고, 필요 시 보호 조치를 즉시 요청하세요."
        paras[0] = (paras[0] + " " + supplement).strip()

    return "\n\n".join(paras)


# 관련법률 중복 제거해주는 함수  
 # 법률명 정규화 (괄호·주석 제거)
def _normalize_law_name(law: str) -> str:
    """
    법률명 + 조문번호만 남기고 괄호/주석은 제거
    예: '민원처리법 제23조 (3회 이상 반복 시 종결)' → '민원처리법 제23조'
    """
    if not law:
        return ""
    return re.sub(r"\s*\(.*?\)", "", law).strip()

# 유형, 법률 중복 항목을 제거 
# sourcePages 후처리 (중복 제거, 최대 3개 유지)
def _post_filter_sources(sources, limit=3):
    """
    - 법률명만 기준으로 중복 제거 (유형이 달라도 같은 법률이면 1개만)
    - 지침/가이드 등 비법률도 허용 (요구사항 반영)
    - 괄호 설명 제거(normalize) + ; , 로 묶인 항목 분할
    - 최대 limit개 유지
    """
    out = []
    seen_laws = set()

    for e in sources or []:
        typ = (e.get("유형") or "").strip()
        raw_law = (e.get("관련법률") or "").strip()
        if not typ or not raw_law or raw_law == "없음":
            continue

        # 여러 개 한 줄일 수 있으니 분할
        for lw in [x.strip() for x in re.split(r"[;,]", raw_law) if x.strip()]:
            norm = _normalize_law_name(lw)  # 괄호/주석 제거
            key = norm.lower()
            if not norm:
                continue
            # ✅ 법률 기준으로 dedup (유형은 달라도 같은 법률이면 skip)
            if key in seen_laws:
                continue
            seen_laws.add(key)
            out.append({"유형": typ, "관련법률": norm})
            if len(out) >= limit:
                return out

    return out


# 유형-관련법률 쌍 단위로 묶어서 줄바꿈 사이에 넣어주는 함수
def format_sourcepages_for_answer(sources: list[dict]) -> str:
    if not sources:
        return ""

    blocks = []
    for e in sources:
        t = e.get("유형", "").strip()
        l = e.get("관련법률", "").strip()
        if not t or not l:
            continue
        block = f"- 유형: {t}\n- 관련법률: {l}"
        blocks.append(block)

    return "\n\n".join(blocks)


# ✅ 5. 유사 문단 검색 (본문+메타데이터 포함)
# Pinecone에서 query와 유사 문단 검색 후 context와 sourcePages 반환
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
            f"📌 **유형:** {typ}\n"
            f"📖 본문: {meta.get('본문', '')}\n"
            f"⚖ **관련 법률**: {law}\n"
            f"📝 요약: {meta.get('요약', '')}\n"
        )
        # ✅ 최종 JSON에서는 '관련법률'(띄어쓰기 없음)
        # ✅ '없음'은 제외해 sourcePages 정합성 보장
        if law and law != "없음":
            law_norm = _normalize_law_name(law)  # ✅ 추가: 괄호·주석 제거
            source_pages.append({"유형": typ, "관련법률": law_norm})

    return "\n---\n".join(context_blocks), source_pages

# ✅ 6. GPT 스트리밍 + JSON 응답 (키워드 기반 법률을 sourcePages 1차 반영)
@router.post("/stream")
async def stream_chat(query: Query):
    # ✅ 0) 일상 대화면 즉시 SSE로 응답하고 종료 (모델/RAG 호출 없이)
    if is_smalltalk(query.question):
        async def smalltalk_events():
            payload = {"answer": smalltalk_reply(query.question), "sourcePages": []}
            yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [END]\n\n"
        return EventSourceResponse(smalltalk_events())
    
    # RAG
    context, source_pages_rag = retrieve_context(query.question)

    # 1차: 키워드 기반 법률쌍
    source_pages_keywords = keyword_pairs_first(query.question)

    # 프롬프트
    prompt = f"""
너는 악성민원 대응 및 관련 법률 상담을 도와주는 AI야.
아래 참고 자료를 바탕으로 사용자의 질문에 대해 자연스럽고 자세한 문장으로 답변해줘.

반드시 JSON으로만 출력하고, 코드 블록은 쓰지 마. 모든 출력은 자연스러운 한글이어야 해.
영어 토큰(예: TYPE), 자리표시자(예: {{유형}})는 절대 사용하지 마. 한국 법령 기준으로 설명해.

- answer는 **정확히 2문단**으로 작성:
  (1문단) 즉시 취해야 할 구체적 조치(보고·기록·심리안정·차단/선종료 기준 등)와 실무 팁을 **4~6문장**으로 서술.
  (2문단) **"당신이 상담한 내용은 ‘{{유형명}}’에 해당할 수 있으며, 관련 법률로는 ‘{{법률명 조문번호}}’가 있습니다."**로 시작.
         이어서 **각 법률마다 1줄**로 핵심 적용 취지를 덧붙여 설명
         (예: 성폭력범죄의 처벌 등에 관한 특례법 제13조(통신매체이용음란): 통신수단으로 성적 수치심을 유발하는 행위를 처벌).
- sourcePages: 아래 참고자료 및 네 추론에 따라 '유형'과 '관련법률'만 배열로 정리(법률·조문 위주).


💡 참고 자료만으로 충분하지 않은 경우의 규칙:
- 만약 아래 참고 자료에서 사용자의 질문과 관련된 유형/법률 정보를 충분히 찾지 못하더라도,
  네가 가진 일반 지식에 기반해 적절한 악성민원 유형과 관련 법률(또는 지침)을 **추론**해서
  answer와 sourcePages에 **함께 포함**해줘.
- 다만 확실하지 않은 경우에는 "해당될 수 있습니다", "관련될 수 있습니다"처럼 **완곡한 표현**을 사용해.
- 법률·조문을 기재할 땐 명칭과 조문 번호를 함께 적어줘. (예: 성폭력범죄의 처벌 등에 관한 특례법 제13조)


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
                "content": ("너는 악성민원 대응 가이드 및 관련 법률 문서를 기반으로 상담하는 전문가 AI다."
                            "반드시 JSON 형식으로만 출력하고, 코드 블록이나 부가 설명은 절대 하지 마."
                            "RAG(참고자료)에서 충분한 정보가 없을 경우에도 일반 지식으로 합리적 추론을 하되, "
                            "불확실한 부분은 단정하지 말고 '관련될 수 있습니다' 등 완곡 표현을 사용해."
                )
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
                    # ✅ 추가: 관련법률 정규화
                    model_sources = [
                        {"유형": ms["유형"], "관련법률": _normalize_law_name(ms["관련법률"])}
                        for ms in model_sources
                    ]

        except Exception:
            pass

        # 병합 규칙: 키워드(1차) → 모델 sourcePages → RAG sourcePages
        final_sources = _merge_sources(source_pages_keywords, model_sources, source_pages_rag)
        
        # ✅ 후처리: 비법률/없음 제거 + 최대 3개 제한
        final_sources = _post_filter_sources(final_sources, limit=3)
        
        # ✅ answer 2문단/시작문장/요약 강제 보정
        final_answer = ensure_two_paragraphs(model_answer, final_sources)

        payload = {"answer": final_answer, "sourcePages": final_sources, "sourcePagesText": format_sourcepages_for_answer(final_sources)}
        yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [END]\n\n"

    return EventSourceResponse(event_generator())
