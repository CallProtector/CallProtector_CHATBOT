import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from pinecone import Pinecone

router = APIRouter()

# ✅ 1. 환경 변수 로드
load_dotenv()

# ✅ 2. FastAPI 앱 초기화
app = FastAPI()

# ✅ 3. OpenAI & Pinecone 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("legal-guideline")

# ✅ 4. 요청 모델 정의
class Query(BaseModel):
    session_id: int
    question: str

# ✅ 5. 유사 문단 검색 (본문+메타데이터 포함)
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
            f"📌 유형: {meta.get('유형', '없음')}\n"
            f"📖 본문: {meta.get('본문', '')}\n"
            f"⚖ 관련 법률: {meta.get('관련 법률', '없음')}\n"
            f"📝 요약: {meta.get('요약', '')}\n"
        )
        source_pages.append({
            "유형": meta.get("유형", "없음"),
            "관련법률": meta.get("관련 법률", "없음")
        })

    return "\n---\n".join(context_blocks), source_pages

# ✅ 6. GPT 스트리밍 + JSON 응답
@app.post("/stream")
async def stream_chat(query: Query):
    context, source_pages = retrieve_context(query.question)
    source_hint = json.dumps(source_pages, ensure_ascii=False, indent=2)

    prompt = f"""
너는 악성민원 대응 및 관련 법률 상담을 도와주는 AI야.
아래 참고 자료를 바탕으로 사용자의 질문에 대해 자연스럽고 자세한 문장으로 답변해줘.

다음 형식의 JSON으로만 출력하고, 코드 블록은 사용하지 마. 모든 출력은 올바른 띄어쓰기를 가진 자연스러운 한글이어야 해.

- answer: 두 문단으로 작성
    1. 사용자의 질문에 대한 일반적이고 자연스러운 답변
    2. "당신이 상담한 내용은 ~유형에 포함되며, 관련 법률로는 ~가 있습니다." 형식으로 설명

    💡 만약 '민원처리법 제23조'와 같은 법률 조항이 등장하면,
       해당 조항의 주요 내용을 한두 문장으로 요약해줘.
       예: "민원처리법 제23조를 보면, 행정기관은 민원 처리 결과를 문서로 통지하고 지연 시 사유와 예정일을 알려야 한다고 규정하고 있습니다."

- sourcePages: 아래 참고자료의 '유형'과 '관련법률'만 배열 형태로 정리

출력 예시:
{{
  "answer": "사용자의 민원은 반복적인 요청과 장시간 통화로 이어졌습니다. 이는 상담사의 대응 효율을 떨어뜨릴 수 있어 적절한 조치가 필요합니다.\n\n당신이 상담한 내용은 '반복 민원' 유형에 해당하며, 관련 법률로는 '국민권익위원회 상담사 보호 지침'이 있습니다.",
  "sourcePages": [
    {{
      "유형": "반복 민원",
      "관련법률": "국민권익위원회 상담사 보호 지침"
    }}
  ]
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
                "content": "너는 악성민원 대응 가이드 및 관련 법률 문서를 기반으로 상담하는 전문가 AI야. 반드시 JSON 형식으로만 출력하고, 코드 블록이나 부가 설명은 절대 하지 마."
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

        yield f"data: [JSON]{full_response}\n\n"
        yield "data: [END]\n\n"

    return EventSourceResponse(event_generator())
