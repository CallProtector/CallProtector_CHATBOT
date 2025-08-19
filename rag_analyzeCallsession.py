from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone
import os
import json
import asyncio
from dotenv import load_dotenv
import re 

# ✅ 환경 변수 로드
load_dotenv()

router = APIRouter()

# ✅ OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# 오레곤(us-west-2) 인덱스 사용
index_name = os.getenv("PINECONE_INDEX", "legal-guideline-usw2")
index = pc.Index(index_name)


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
    "감정노동 종사자 건강보호 가이드": "감정노동으로 인한 피해를 예방하고 종사자 보호 기준을 제시합니다.",
    "서울특별시 감정노동 종사자 권리보호 조례 제2조": "감정노동 종사자의 권리 보호 및 기관의 책무를 규정합니다."
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




def _normalize_law_name(law: str) -> str:
    """
    법률명 + 조문번호만 남기고 괄호/주석은 제거.
    예: '민원처리법 제23조 (3회 이상 반복 시 종결)' → '민원처리법 제23조'
    """
    if not law:
        return ""
    return re.sub(r"\s*\(.*?\)", "", law).strip()

def _post_filter_sources(sources, limit=3):
    """
    - '관련법률'이 '없음'이거나 빈 값이면 제외
    - ';' 또는 ',' 로 묶인 다중 법률 분할 후 각각 정규화
    - 같은 법률(정규화 기준) 중복 제거 (유형 달라도 법률이 같으면 1개만)
    - 최대 limit개로 제한
    """
    out = []
    seen_laws = set()

    for e in sources or []:
        typ = (e.get("유형") or "").strip()
        raw_law = (e.get("관련법률") or "").strip()
        if not typ or not raw_law or raw_law == "없음":
            continue

        # 여러 개가 한 줄에 들어오는 경우 분할
        for lw in [x.strip() for x in re.split(r"[;,]", raw_law) if x.strip()]:
            norm = _normalize_law_name(lw)
            if not norm:
                continue
            key = norm.lower()
            if key in seen_laws:
                continue
            seen_laws.add(key)
            out.append({"유형": typ, "관련법률": norm})
            if len(out) >= limit:
                return out

    return out

# ✅ 벡터 검색 함수 정의
def retrieve_context(query: str, top_k: int = 5):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"  # ✅ 더 빠르고 저렴
    ).data[0].embedding

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    context_blocks = []
    source_pages = []
    for match in results["matches"]:
        meta = match["metadata"]
        context_blocks.append(
            f"\U0001f4cc 유형: {meta.get('유형', '없음')}\n"
            f"\U0001f4d6 본문: {meta.get('본문', '')}\n"
            f"⚖ 관련 법률: {meta.get('관련 법률', '없음')}\n"
            f"\U0001f4dd 요약: {meta.get('요약', '')}\n"
        )
        source_pages.append({
            "유형": meta.get("유형", "없음"),
            "관련법률": meta.get("관련 법률", "없음")
        })

    return "\n---\n".join(context_blocks), source_pages



# ---------- 🧠 분석 엔드포인트 ----------

@router.post("/analyze")
async def analyze_call_session(request: Request):
    body = await request.json()
    session_id = body.get("sessionId")
    user_id = body.get("userId")
    scripts = body.get("scripts", [])

    if not scripts:
        return EventSourceResponse(content=("data: 세션에 스크립트 없음\n\n",), status_code=400)

    # ✅ 통화 내용 추출
    context_dialogue = "\n".join(f"{s['speaker']}: {s['text']}" for s in scripts)

    # ✅ RAG 질의
    question = (
        "다음 상담 내용에서 고객이 성희롱, 폭언, 협박 등의 발언을 했다면 "
        "어떤 법률 조항(법률명 + 조문번호 포함)이 적용될 수 있으며, "
        "어떻게 대응해야 하는지 알려줘.\n\n"
        f"{context_dialogue}"
    )

    # ✅ RAG 검색
    rag_context, source_pages_rag = retrieve_context(question)

    # ✅ 추가 법률 정보 (UI 참고용)
    additional_laws = ""
    if "성희롱" in question:
        additional_laws += "\n📚 성희롱 관련 법률:\n- 성폭력범죄의 처벌 등에 관한 특례법 제13조: 2년 이하 징역 또는 2천만원 이하 벌금"
    if any(k in question for k in ["욕설", "협박", "폭언"]):
        additional_laws += "\n📚 욕설·협박·폭언 관련 법률:\n- 형법 제283조(협박): 3년 이하 징역 또는 500만원 이하 벌금\n- 형법 제260조(폭행): 2년 이하 징역 또는 500만원 이하 벌금"
    if any(k in question for k in ["모욕", "명예훼손", "폭언"]):
        additional_laws += "\n📚 명예훼손·모욕·폭언 관련 법률:\n- 형법 제307조(명예훼손): 2년 이하 징역 또는 500만원 이하 벌금\n- 형법 제311조(모욕): 1년 이하 징역 또는 200만원 이하 벌금"
    if "업무방해" in question:
        additional_laws += "\n📚 업무방해 관련 법률:\n- 형법 제314조(업무방해): 5년 이하 징역 또는 1천5백만원 이하 벌금"
    if "강요" in question:
        additional_laws += "\n📚 강요 관련 법률:\n- 형법 제324조(강요): 5년 이하 징역 또는 3천만원 이하 벌금"
    if any(k in question for k in ["장난전화", "괴롭힘", "반복적인 민원"]):
        additional_laws += "\n📚 장난전화/경범(강성 민원) 관련 법률:\n- 경범죄처벌법 제3조 제1항 제40호: 10만원 이하 벌금, 구류, 과료"
    if "스토킹" in question:
        additional_laws += "\n📚 스토킹 관련 법률:\n- 스토킹범죄의 처벌 등에 관한 법률 제18조 제1항: 3년 이하 징역 또는 3천만원 이하 벌금"

    if additional_laws:
        rag_context += "\n---\n" + additional_laws

    # ✅ 프롬프트 (원본 그대로 유지)
    prompt = f"""
너는 악성민원 대응 및 관련 법률 자문을 돕는 전문가 AI야.

아래 통화 내용을 참고해서 다음 형식에 맞춰 정중하고 구조화된 요약을 생성해줘.

✅ 특히 **적용 가능한 법률**에는 반드시 '법률명 + 조문번호 + 조문명'을 포함해서 작성하고,
   각 법률이 어떤 악성 발언 유형(예: 성희롱, 모욕, 명예훼손 등)에 대응되는지 간단히 설명해줘.

✅ 또한 Markdown 문법과 아이콘을 포함하고, 출력은 자연스럽고 띄어쓰기가 올바른 한국어 문장으로 작성해줘.

[응답 형식 예시]
안녕하세요 000님, 방금 상담 중 고객으로부터 폭언이나 성희롱 발언을 받으셨네요. 관련 법률과 대응 방법을 안내해드릴게요.

📜 **적용 가능한 법률:**
- **형법 제311조(모욕죄)**: 상대방을 공개적으로 모욕할 경우 적용됩니다.
- **성폭력범죄의 처벌 등에 관한 특례법 제13조(통신매체를 이용한 음란행위)**: 성적 수치심을 유발하는 발언이 있을 경우 적용될 수 있습니다.

⚖️ **대응 방법:**
1. **사내 대응 절차**
   - 📌 **1차 조치**: 고객 발언이 욕설·성희롱·협박에 해당될 경우 즉시 ARS 경고멘트를 송출하거나 통화 종료 권한을 행사할 수 있습니다.
   - 📝 **2차 조치**: 통화 종료 후, 소속 부서장에게 상황을 **보고**하고, 상담사 보호를 위한 **사내 대응 매뉴얼에 따라 악성민원 등록**을 요청하세요.
   - 🧾 **3차 조치**: 필요시 악성민원 전담관리자가 해당 고객의 **재통화 차단**, **주의 고객 등록**, **전담 응대 팀 이관** 등을 검토할 수 있습니다.
   - 🧠 **상담 지원**: 정신적 충격이 있는 경우, **EAP 프로그램(심리상담/내부상담센터)** 등을 통해 보호 조치를 받을 수 있습니다.

2. **법적 조치**
   - 🕵️ **내용 기록**: 폭언, 성희롱, 협박 등이 있었다면 해당 발언 내용을 녹취 및 대화 로그로 보관하고, **상세 보고서 작성**을 권장합니다.
   - 💼 **사내 법무팀/감사팀 협조 요청**: 반복적이거나 악의적인 사례는 법무팀과 협의해 **경고장 발송**, **법률 자문**, **형사고발 여부 검토** 등을 진행할 수 있습니다.
   - ⚖️ **형사고소 및 민사청구**: 실제 피해 발생 시에는 모욕죄·명예훼손·강요·스토킹 등으로 고소가 가능하며, 정신적 피해에 따른 **위자료 청구**도 고려할 수 있습니다.

➕ 추가로 도움이 필요하시면 언제든 말씀해주세요!

[❗단, 문제가 되지 않는 평범한 통화라면 "법적 조치 대상은 아니며 일반 민원 응대로 판단됩니다."로 간단히 응답해줘.]

---

# 통화 내용:
{context_dialogue}

# 참고 법률 자료:
{rag_context}
"""

    # ✅ GPT 스트리밍 응답
    async def event_generator():
        full_response = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    yield f"{delta}\n\n"

            # --- 병합 로직 (callchat_stream과 동일) ---
            # 1) 키워드 기반
            kw_sources = keyword_pairs_first(context_dialogue + "\n" + question)

            # 2) 모델 소스 (JSON 파싱 시도)
            model_sources = []
            try:
                parsed = json.loads(full_response)
                if isinstance(parsed, dict) and isinstance(parsed.get("sourcePages"), list):
                    model_sources = [
                        {"유형": (e.get("유형") or "").strip(),
                         "관련법률": _normalize_law_name((e.get("관련법률") or "").strip())}
                        for e in parsed["sourcePages"] if isinstance(e, dict)
                    ]
            except Exception:
                pass

            # 3) RAG 소스
            rag_sources = source_pages_rag
            
            # ✅ 평범한 통화 여부 체크
            if not kw_sources and not model_sources and not rag_sources:
                final_answer = (
                    "안녕하세요 고객님, 방금 통화 중에 발생한 상황에 대해 처리 방법과 관련 법률을 안내해드리겠습니다.\n\n"
                    "현재 통화 내용에서는 특별히 문제가 되는 발언이 발견되지 않았습니다. "
                    "따라서 본 건은 법적 조치 대상은 아니며 일반 민원 응대로 판단됩니다.\n\n"
                    "➕ 추가로 도움이 필요하시면 언제든 말씀해주세요!"
                )
                payload = {"answer": final_answer, "sourcePages": []}
                yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [END]\n\n"
                return  # ✅ 여기서 종료


            # 4) 최종 병합 (kw → model → rag)
            merged = kw_sources + model_sources + rag_sources

            # 5) 후처리 (dedup/정규화/최대 3개)
            final_sources = _post_filter_sources(merged, limit=3)

            # 6) answer 보정 (두 번째 문단만 교체)
            final_answer = _ensure_two_paragraphs(full_response, final_sources)

            payload = {"answer": final_answer, "sourcePages": final_sources}
            yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [END]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [END]\n\n"

    return EventSourceResponse(event_generator())
