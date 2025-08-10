from fastapi import FastAPI, Request
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone
import os
import json
import asyncio
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

app = FastAPI()

# ✅ OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-guideline")

# ✅ 벡터 검색 함수 정의
def retrieve_context(query: str, top_k: int = 5):
    embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-ada-002"
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


@app.post("/api/chatbot/analyze")
async def analyze_call_session(request: Request):
    body = await request.json()
    session_id = body.get("sessionId")
    user_id = body.get("userId")
    scripts = body.get("scripts", [])

    if not scripts:
        return EventSourceResponse(content=("data: 세션에 스크립트 없음\n\n",), status_code=400)

    # ✅ 통화 내용 추출
    context_dialogue = "\n".join(f"{s['speaker']}: {s['text']}" for s in scripts)

    # ✅ RAG용 질의 생성
    question = f"다음 상담 내용에서 고객이 성희롱, 폭언, 협박 등의 발언을 했다면 어떤 법률 조항(법률명 + 조문번호 포함)이 적용될 수 있으며, 어떻게 대응해야 하는지 알려줘.\n\n{context_dialogue}"

    # ✅ RAG 문단 검색
    rag_context, source_pages = retrieve_context(question)

    # ✅ 추가 법률 정보 삽입
    additional_laws = ""

    if "성희롱" in question:
        additional_laws += "\n📚 성희롱 관련 법률:\n- 성폭력범죄의 처벌 등에 관한 특례법 제13조: 2년 이하 징역 또는 2천만원 이하 벌금"

    if "욕설" in question or "협박" in question:
        additional_laws += "\n📚 욕설·협박 관련 법률:\n- 형법 제283조(협박): 3년 이하 징역 또는 500만원 이하 벌금\n- 형법 제260조(폭행): 2년 이하 징역 또는 500만원 이하 벌금"

    if "모욕" in question or "명예훼손" in question:
        additional_laws += "\n📚 명예훼손·모욕 관련 법률:\n- 형법 제307조(명예훼손): 2년 이하 징역 또는 500만원 이하 벌금\n- 형법 제311조(모욕): 1년 이하 징역 또는 200만원 이하 벌금"

    if "업무방해" in question:
        additional_laws += "\n📚 업무방해 관련 법률:\n- 형법 제314조(업무방해): 5년 이하 징역 또는 1천5백만원 이하 벌금"

    if "강요" in question:
        additional_laws += "\n📚 강요 관련 법률:\n- 형법 제324조(강요): 5년 이하 징역 또는 3천만원 이하 벌금"

    if "장난전화" in question or "괴롭힘" in question:
        additional_laws += "\n📚 장난전화 관련 법률:\n- 경범죄처벌법 제3조 제1항 제40호: 10만원 이하 벌금, 구류, 과료"

    if "스토킹" in question:
        additional_laws += "\n📚 스토킹 관련 법률:\n- 스토킹범죄의 처벌 등에 관한 법률 제18조 제1항: 3년 이하 징역 또는 3천만원 이하 벌금"

    if additional_laws:
        rag_context += "\n---\n" + additional_laws


    # ✅ 프롬프트 작성
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
                    await asyncio.sleep(0.01)

            # 이전 ver 
            #yield f"data: [JSON]{full_response}\n\n"
            # 수정 ver
            payload = {"answer": full_response, "sourcePages": source_pages}
            yield f"data: [JSON]{json.dumps(payload, ensure_ascii=False)}\n\n"
            
            yield "data: [END]\n\n"
        except Exception as e:
            # 에러는 에러로만 통지 ( JSON 변환 이딴 거 X)
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [END]\n\n"

    return EventSourceResponse(event_generator())
